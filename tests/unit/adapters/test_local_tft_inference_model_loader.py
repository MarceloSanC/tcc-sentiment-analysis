from __future__ import annotations

import json
import logging
import pickle
import sys
import types

from pathlib import Path

import pytest

from src.adapters.local_tft_inference_model_loader import LocalTFTInferenceModelLoader


class _FakeTFTModel:
    def eval(self) -> _FakeTFTModel:
        return self


def _install_fake_pytorch_forecasting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("pytorch_forecasting")

    class _FakeTemporalFusionTransformer:
        @staticmethod
        def load_from_checkpoint(path: str):
            return _FakeTFTModel()

    fake_module.TemporalFusionTransformer = _FakeTemporalFusionTransformer
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_module)


def _write_valid_artifact_bundle(
    version_dir: Path,
    *,
    version: str,
    training_run_id: str | None = None,
) -> None:
    (version_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (version_dir / "checkpoints" / "best.ckpt").write_bytes(b"ckpt")
    metadata = {"asset_id": "AAPL", "version": version}
    if training_run_id is not None:
        metadata["training_run_id"] = training_run_id
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (version_dir / "config.json").write_text(
        json.dumps({"training_config": {"max_encoder_length": 3, "max_prediction_length": 1}}),
        encoding="utf-8",
    )
    (version_dir / "features.json").write_text(
        json.dumps({"features_used": ["close"]}),
        encoding="utf-8",
    )


def test_loader_rejects_invalid_model_version_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_pytorch_forecasting(monkeypatch)

    version_dir = tmp_path / "20260303_120000_B"
    _write_valid_artifact_bundle(version_dir, version="invalid_version")

    loader = LocalTFTInferenceModelLoader()
    with pytest.raises(ValueError, match="invalid `version`"):
        loader.load(version_dir)


def test_loader_requires_directory_name_match_metadata_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_pytorch_forecasting(monkeypatch)

    version_dir = tmp_path / "20260303_120000_B"
    _write_valid_artifact_bundle(version_dir, version="20260303_120001_B")

    loader = LocalTFTInferenceModelLoader()
    with pytest.raises(ValueError, match="must match metadata.json `version`"):
        loader.load(version_dir)


def test_loader_loads_dataset_parameters_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_pytorch_forecasting(monkeypatch)

    version = "20260303_120000_B"
    version_dir = tmp_path / version
    _write_valid_artifact_bundle(version_dir, version=version)
    with (version_dir / "dataset_parameters.pkl").open("wb") as fp:
        pickle.dump({"time_idx": "time_idx", "group_ids": ["asset_id"]}, fp)

    loader = LocalTFTInferenceModelLoader()
    bundle = loader.load(version_dir)
    assert bundle.dataset_parameters.get("time_idx") == "time_idx"
    assert bundle.dataset_parameters.get("group_ids") == ["asset_id"]


def test_inference_loader_reads_training_run_id_from_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_pytorch_forecasting(monkeypatch)

    version = "20260303_120000_B"
    version_dir = tmp_path / version
    _write_valid_artifact_bundle(
        version_dir,
        version=version,
        training_run_id="train_run_abc",
    )

    loader = LocalTFTInferenceModelLoader()
    bundle = loader.load(version_dir)

    assert bundle.training_run_id == "train_run_abc"


def test_inference_loader_returns_none_for_legacy_metadata_without_training_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_fake_pytorch_forecasting(monkeypatch)
    caplog.set_level(logging.WARNING)

    version = "20260303_120000_B"
    version_dir = tmp_path / version
    _write_valid_artifact_bundle(version_dir, version=version)
    LocalTFTInferenceModelLoader._warned_legacy_metadata_paths.clear()

    loader = LocalTFTInferenceModelLoader()
    bundle = loader.load(version_dir)
    second_bundle = loader.load(version_dir)

    assert bundle.training_run_id is None
    assert second_bundle.training_run_id is None
    warnings = [
        r for r in caplog.records
        if "missing training_run_id" in r.message
    ]
    assert len(warnings) == 1


def test_loader_rejects_non_dict_dataset_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_pytorch_forecasting(monkeypatch)

    version = "20260303_120000_B"
    version_dir = tmp_path / version
    _write_valid_artifact_bundle(version_dir, version=version)
    with (version_dir / "dataset_parameters.pkl").open("wb") as fp:
        pickle.dump(["not", "a", "dict"], fp)

    loader = LocalTFTInferenceModelLoader()
    with pytest.raises(ValueError, match="INFER_DATASET_SPEC_INVALID_TYPE"):
        loader.load(version_dir)
