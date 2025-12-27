# tests/integration/test_feature_engineering_pipeline.py
from src.use_cases.feature_engineering_use_case import FeatureEngineeringUseCase


def test_feature_engineering_pipeline_runs(tmp_path):
    use_case = FeatureEngineeringUseCase(
        input_dir="data/raw",
        output_dir=tmp_path,
    )

    use_case.execute("AAPL")

    output_files = list(tmp_path.glob("features_*.parquet"))
    assert len(output_files) == 1
