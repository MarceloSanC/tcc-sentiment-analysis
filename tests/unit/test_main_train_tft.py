from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src import main_train_tft


def _base_args(config_json: str | None = None, features: str | None = None) -> Namespace:
    return Namespace(
        asset="AAPL",
        features=features,
        config_json=config_json,
        models_dir=None,
        max_encoder_length=None,
        max_prediction_length=None,
        batch_size=None,
        max_epochs=None,
        learning_rate=None,
        hidden_size=None,
        attention_head_size=None,
        dropout=None,
        hidden_continuous_size=None,
        seed=None,
        early_stopping_patience=None,
        early_stopping_min_delta=None,
        train_start=None,
        train_end=None,
        val_start=None,
        val_end=None,
        test_start=None,
        test_end=None,
        run_ablation=False,
    )


def test_main_train_tft_reads_json_config(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "train_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "features": ["BASELINE_FEATURES", "sentiment_score"],
                "run_ablation": True,
                "training_config": {"max_epochs": 7, "dropout": 0.2},
                "split_config": {
                    "train_start": "20120101",
                    "train_end": "20201231",
                    "val_start": "20210101",
                    "val_end": "20221231",
                    "test_start": "20230101",
                    "test_end": "20251231",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main_train_tft, "load_data_paths", lambda: {"dataset_tft": tmp_path, "models": tmp_path})
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: _base_args(config_json=str(cfg_path)))
    monkeypatch.setattr(main_train_tft, "ParquetTFTDatasetRepository", lambda **_: object())
    monkeypatch.setattr(main_train_tft, "LocalTFTModelRepository", lambda **_: object())
    monkeypatch.setattr(main_train_tft, "PytorchForecastingTFTTrainer", lambda: object())

    captured: dict = {}

    class _FakeUseCase:
        def __init__(self, **kwargs):
            pass

        def execute(self, asset_id: str, **kwargs):
            captured["asset_id"] = asset_id
            captured.update(kwargs)
            return type(
                "_R",
                (),
                {"asset_id": asset_id, "version": "v", "artifacts_dir": "d", "metrics": {"rmse": 1.0}},
            )()

    monkeypatch.setattr(main_train_tft, "TrainTFTModelUseCase", _FakeUseCase)
    main_train_tft.main()

    assert captured["asset_id"] == "AAPL"
    assert captured["features"] == ["BASELINE_FEATURES", "sentiment_score"]
    assert captured["run_ablation"] is True
    assert captured["training_config"]["max_epochs"] == 7
    assert captured["training_config"]["dropout"] == 0.2
    assert captured["split_config"]["train_start"] == "20120101"


def test_main_train_tft_cli_overrides_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "train_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "features": ["BASELINE_FEATURES"],
                "run_ablation": False,
                "training_config": {"max_epochs": 7},
            }
        ),
        encoding="utf-8",
    )

    args = _base_args(config_json=str(cfg_path), features="close,open")
    args.max_epochs = 13
    args.run_ablation = True

    monkeypatch.setattr(main_train_tft, "load_data_paths", lambda: {"dataset_tft": tmp_path, "models": tmp_path})
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: args)
    monkeypatch.setattr(main_train_tft, "ParquetTFTDatasetRepository", lambda **_: object())
    monkeypatch.setattr(main_train_tft, "LocalTFTModelRepository", lambda **_: object())
    monkeypatch.setattr(main_train_tft, "PytorchForecastingTFTTrainer", lambda: object())

    captured: dict = {}

    class _FakeUseCase:
        def __init__(self, **kwargs):
            pass

        def execute(self, asset_id: str, **kwargs):
            captured.update(kwargs)
            return type(
                "_R",
                (),
                {"asset_id": asset_id, "version": "v", "artifacts_dir": "d", "metrics": {"rmse": 1.0}},
            )()

    monkeypatch.setattr(main_train_tft, "TrainTFTModelUseCase", _FakeUseCase)
    main_train_tft.main()

    assert captured["features"] == ["close", "open"]
    assert captured["training_config"]["max_epochs"] == 13
    assert captured["run_ablation"] is True


def test_main_train_tft_raises_on_missing_json_path(monkeypatch, tmp_path: Path) -> None:
    args = _base_args(config_json=str(tmp_path / "missing.json"))
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="Config JSON not found"):
        main_train_tft.main()


def test_main_train_tft_raises_on_invalid_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text("{invalid", encoding="utf-8")
    args = _base_args(config_json=str(cfg_path))
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="Invalid JSON"):
        main_train_tft.main()


def test_main_train_tft_raises_on_non_object_json_root(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_root.json"
    cfg_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    args = _base_args(config_json=str(cfg_path))
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="root must be an object"):
        main_train_tft.main()


def test_main_train_tft_features_csv_string_in_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "train_cfg.json"
    cfg_path.write_text(
        json.dumps({"features": "BASELINE_FEATURES,sentiment_score"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(main_train_tft, "load_data_paths", lambda: {"dataset_tft": tmp_path, "models": tmp_path})
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: _base_args(config_json=str(cfg_path)))
    monkeypatch.setattr(main_train_tft, "ParquetTFTDatasetRepository", lambda **_: object())
    monkeypatch.setattr(main_train_tft, "LocalTFTModelRepository", lambda **_: object())
    monkeypatch.setattr(main_train_tft, "PytorchForecastingTFTTrainer", lambda: object())

    captured: dict = {}

    class _FakeUseCase:
        def __init__(self, **kwargs):
            pass

        def execute(self, asset_id: str, **kwargs):
            captured.update(kwargs)
            return type(
                "_R",
                (),
                {"asset_id": asset_id, "version": "v", "artifacts_dir": "d", "metrics": {"rmse": 1.0}},
            )()

    monkeypatch.setattr(main_train_tft, "TrainTFTModelUseCase", _FakeUseCase)
    main_train_tft.main()

    assert captured["features"] == ["BASELINE_FEATURES", "sentiment_score"]


def test_main_train_tft_range_validation_raises(monkeypatch, tmp_path: Path) -> None:
    args = _base_args()
    args.dropout = 1.5
    monkeypatch.setattr(main_train_tft, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="dropout"):
        main_train_tft.main()
