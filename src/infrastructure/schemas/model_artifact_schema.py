# src/infrastructure/schemas/model_artifact_schema.py

MODEL_METADATA_FIELDS = {
    "model_type",
    "asset_id",
    "created_at",
    "training_window",
    "features_used",
    "scaler_type",
    "version",
}

TFT_TRAINING_DEFAULTS = {
    "max_encoder_length": 60,
    "max_prediction_length": 1,
    "batch_size": 64,
    "max_epochs": 20,
    "learning_rate": 1e-3,
    "hidden_size": 16,
    "attention_head_size": 2,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "seed": 42,
}

TFT_SPLIT_DEFAULTS = {
    "train_start": "20100101",
    "train_end": "20221231",
    "val_start": "20230101",
    "val_end": "20241231",
    "test_start": "20250101",
    "test_end": "20251231",
}
