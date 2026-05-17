from __future__ import annotations

import json
import logging
import pickle
import re
import warnings

from pathlib import Path
from typing import Any, ClassVar

from src.interfaces.tft_inference_model_loader import (
    LoadedTFTInferenceModel,
    TFTInferenceModelLoader,
)
from src.utils.path_policy import to_project_relative


class LocalTFTInferenceModelLoader(TFTInferenceModelLoader):
    """
    Loads a TFT model artifact directory produced by LocalTFTModelRepository.
    """
    _MODEL_VERSION_PATTERN = re.compile(r"^\d{8}_\d{6}_[A-Z0-9]+$")
    _GPU_DESERIALIZE_TOKENS = ("No HIP GPUs are available", "CUDA", "cuda")
    _logger = logging.getLogger(__name__)
    _warned_legacy_metadata_paths: ClassVar[set[str]] = set()

    @staticmethod
    def _load_checkpoint_on_cpu_fallback(
        checkpoint_path: Path,
        temporal_fusion_transformer_cls: Any,
    ) -> Any:
        import torch

        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid checkpoint payload: expected dict root object."
            )
        hyper_parameters = checkpoint.get("hyper_parameters")
        if not isinstance(hyper_parameters, dict):
            raise ValueError(
                "Invalid checkpoint payload: missing dict `hyper_parameters`."
            )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(
                "Invalid checkpoint payload: missing dict `state_dict`."
            )
        model = temporal_fusion_transformer_cls(**hyper_parameters)
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _sanitize_torchmetrics_device_to_cpu(model: Any) -> None:
        import torch

        try:
            from torchmetrics import Metric
        except Exception:
            return

        if not hasattr(model, "modules"):
            return

        for module in model.modules():
            if isinstance(module, Metric):
                try:
                    module._device = torch.device("cpu")
                except Exception:
                    continue

    @staticmethod
    def _load_json(path: Path, required: bool = True) -> dict[str, Any]:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required model artifact not found: {path.resolve()}")
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path.resolve()}")
        return data

    def load(self, model_dir: str | Path) -> LoadedTFTInferenceModel:
        model_path = Path(model_dir)
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        metadata = self._load_json(model_path / "metadata.json", required=True)
        config = self._load_json(model_path / "config.json", required=True)
        features_cfg = self._load_json(model_path / "features.json", required=True)

        features_used = features_cfg.get("features_used")
        if not isinstance(features_used, list) or not all(
            isinstance(c, str) and c.strip() for c in features_used
        ):
            raise ValueError(
                "Invalid features.json: expected non-empty list in `features_used`."
            )

        checkpoint_path = model_path / "checkpoints" / "best.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Best checkpoint not found. Expected file: "
                f"{checkpoint_path.resolve()}"
            )

        try:
            from pytorch_forecasting import TemporalFusionTransformer
        except Exception as exc:
            raise RuntimeError(
                "pytorch-forecasting is required for TFT inference."
            ) from exc

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Attribute 'loss' is an instance of `nn.Module`.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Attribute 'logging_metrics' is an instance of `nn.Module`.*",
                    category=UserWarning,
                )
                model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
        except Exception as exc:
            message = str(exc)
            if any(token in message for token in self._GPU_DESERIALIZE_TOKENS):
                self._logger.warning(
                    "Checkpoint load failed on GPU path; retrying on CPU",
                    extra={"checkpoint_path": str(checkpoint_path)},
                )
                model = self._load_checkpoint_on_cpu_fallback(
                    checkpoint_path=checkpoint_path,
                    temporal_fusion_transformer_cls=TemporalFusionTransformer,
                )
            else:
                raise
        try:
            import torch
            if not torch.cuda.is_available():
                self._sanitize_torchmetrics_device_to_cpu(model)
        except Exception:
            self._sanitize_torchmetrics_device_to_cpu(model)
        model.eval()

        scalers: dict[str, Any] = {}
        scalers_path = model_path / "scalers.pkl"
        if scalers_path.exists():
            with scalers_path.open("rb") as fp:
                loaded = pickle.load(fp)
            if isinstance(loaded, dict):
                scalers = loaded

        dataset_parameters: dict[str, Any] = {}
        dataset_params_path = model_path / "dataset_parameters.pkl"
        if dataset_params_path.exists():
            with dataset_params_path.open("rb") as fp:
                loaded = pickle.load(fp)
            if not isinstance(loaded, dict):
                raise ValueError(
                    "[INFER_DATASET_SPEC_INVALID_TYPE] dataset_parameters.pkl must contain a dict. "
                    f"model_path={model_path.resolve()}"
                )
            dataset_parameters = loaded

        asset_id = str(metadata.get("asset_id") or "").strip().upper()
        version = str(metadata.get("version") or "").strip()
        if not asset_id:
            raise ValueError("metadata.json missing `asset_id`.")
        if not version:
            raise ValueError("metadata.json missing `version`.")
        if not self._MODEL_VERSION_PATTERN.fullmatch(version):
            raise ValueError(
                "metadata.json has invalid `version`. Expected pattern: YYYYMMDD_HHMMSS_<TAG>."
            )
        if model_path.name != version:
            raise ValueError(
                "Model directory name must match metadata.json `version` for traceability "
                f"(dir={model_path.name}, metadata={version})."
            )
        raw_training_run_id = metadata.get("training_run_id")
        training_run_id = (
            str(raw_training_run_id).strip()
            if raw_training_run_id is not None and str(raw_training_run_id).strip()
            else None
        )
        if training_run_id is None:
            metadata_key = str((model_path / "metadata.json").resolve())
            if metadata_key not in self._warned_legacy_metadata_paths:
                self._warned_legacy_metadata_paths.add(metadata_key)
                self._logger.warning(
                    "Model metadata is missing training_run_id; treating artifact as pre-Stage 10 legacy "
                    "and persisting inference silver run_id as None.",
                    extra={"metadata_path": metadata_key},
                )

        training_config = config.get("training_config")
        if not isinstance(training_config, dict):
            training_config = config

        raw_tokens = config.get("feature_tokens")
        feature_tokens: list[str] = []
        if isinstance(raw_tokens, list):
            feature_tokens = [str(t).strip() for t in raw_tokens if str(t).strip()]
        elif isinstance(raw_tokens, str) and raw_tokens.strip():
            feature_tokens = [t.strip() for t in raw_tokens.split(",") if t.strip()]

        if feature_tokens:
            feature_set_name = "+".join(feature_tokens)
        else:
            feature_set_name = ",".join([c.strip() for c in features_used if c.strip()])

        return LoadedTFTInferenceModel(
            asset_id=asset_id,
            version=version,
            model_dir=Path(to_project_relative(model_path) or str(model_path)),
            model=model,
            feature_cols=[c.strip() for c in features_used if c.strip()],
            feature_set_name=feature_set_name,
            feature_tokens=feature_tokens,
            training_config=training_config,
            training_run_id=training_run_id,
            scalers=scalers,
            dataset_parameters=dataset_parameters,
        )
