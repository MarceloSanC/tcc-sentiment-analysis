from __future__ import annotations

import argparse
import json
import logging
import os

from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.adapters.alpha_vantage_fundamental_fetcher import (
    AlphaVantageFundamentalFetcher,
)
from src.adapters.alpha_vantage_news_fetcher import AlphaVantageNewsFetcher
from src.adapters.finbert_sentiment_model import FinBERTSentimentModel
from src.adapters.local_tft_inference_model_loader import LocalTFTInferenceModelLoader
from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.adapters.parquet_daily_sentiment_repository import (
    ParquetDailySentimentRepository,
)
from src.adapters.parquet_fundamental_repository import ParquetFundamentalRepository
from src.adapters.parquet_news_repository import ParquetNewsRepository
from src.adapters.parquet_scored_news_repository import ParquetScoredNewsRepository
from src.adapters.parquet_technical_indicator_repository import (
    ParquetTechnicalIndicatorRepository,
)
from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.adapters.parquet_tft_inference_repository import ParquetTFTInferenceRepository
from src.adapters.pytorch_forecasting_tft_inference_engine import (
    PytorchForecastingTFTInferenceEngine,
)
from src.adapters.technical_indicator_calculator import TechnicalIndicatorCalculator
from src.adapters.yfinance_candle_fetcher import YFinanceCandleFetcher
from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.domain.time.utc import ensure_utc
from src.use_cases.build_tft_dataset_use_case import BuildTFTDatasetUseCase
from src.use_cases.fetch_candles_use_case import FetchCandlesUseCase
from src.use_cases.fetch_fundamentals_use_case import FetchFundamentalsUseCase
from src.use_cases.fetch_news_use_case import FetchNewsUseCase
from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase
from src.use_cases.run_tft_inference_use_case import RunTFTInferenceUseCase
from src.use_cases.sentiment_feature_engineering_use_case import (
    SentimentFeatureEngineeringUseCase,
)
from src.use_cases.technical_indicator_engineering_use_case import (
    TechnicalIndicatorEngineeringUseCase,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TFT inference using a trained model artifact directory."
    )
    parser.add_argument("--asset", type=str, help="Asset symbol (e.g. AAPL)")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model artifact version directory.",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        help="Path to JSON config. Merge order: defaults <- JSON <- CLI.",
    )
    parser.add_argument("--start", type=str, help="Inference start date (yyyymmdd)")
    parser.add_argument("--end", type=str, help="Inference end date (yyyymmdd)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite inference rows in requested period (by timestamp/model_version).",
    )
    parser.add_argument(
        "--overwrite-on-collision",
        action="store_true",
        help="Overwrite colliding analytics silver rows instead of failing on logical PK collision.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size for dataloader.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Optional dataset_tft base directory override.",
    )
    parser.add_argument(
        "--inference-dir",
        type=str,
        help="Optional inference repository base directory override.",
    )
    parser.add_argument(
        "--auto-refresh",
        action="store_true",
        help=(
            "When dataset_tft lacks requested end-date coverage, run incremental fetch/build "
            "pipelines automatically before inference."
        ),
    )
    parser.add_argument(
        "--allow-missing-quantiles",
        action="store_true",
        help=(
            "Allow inference to complete even when quantile outputs (p10/p50/p90) are missing "
            "for models configured with prediction_mode=quantile."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["rolling", "last_point"],
        help="Inference output mode: rolling (default) or last_point (debug).",
    )
    return parser.parse_args()


def _load_json_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"Config JSON not found: {path}")
    try:
        content = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc
    if not isinstance(content, dict):
        raise ValueError("Config JSON root must be an object")
    return content


def _parse_yyyymmdd(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid date format: {value}. Expected yyyymmdd.") from exc


def _resolve_model_path_input(model_path: str, asset: str, models_root: Path) -> str:
    raw = model_path.strip()
    if not raw:
        return raw

    candidate = Path(raw)
    if candidate.exists():
        return str(candidate)

    # Shorthand support: allow passing only VERSION (e.g., 20260308_180007_BT).
    # Priority follows current default training layout and then legacy layout.
    expanded_candidates = [
        models_root / asset / "runs" / raw,
        models_root / asset / raw,
    ]
    for c in expanded_candidates:
        if c.exists():
            return str(c)

    return raw


def _build_refresh_fn(paths: dict):
    def _refresh_dataset_incremental(
        asset_id: str,
        refresh_start: datetime,
        refresh_end: datetime,
        rebuild_start: datetime,
    ) -> None:
        logger.info(
            "Refreshing source datasets for inference coverage",
            extra={
                "asset": asset_id,
                "refresh_start": refresh_start.isoformat(),
                "refresh_end": refresh_end.isoformat(),
                "rebuild_start": rebuild_start.isoformat(),
            },
        )

        candles_repo = ParquetCandleRepository(output_dir=paths["raw_candles"])
        candles_fetcher = YFinanceCandleFetcher()
        FetchCandlesUseCase(candles_fetcher, candles_repo).execute(
            asset_id,
            refresh_start,
            refresh_end,
        )

        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ALPHAVANTAGE_API_KEY not found in environment; required for dataset refresh."
            )

        news_repo = ParquetNewsRepository(output_dir=paths["news_dataset"])
        news_fetcher = AlphaVantageNewsFetcher(api_key=api_key)
        FetchNewsUseCase(news_fetcher, news_repo).execute(
            asset_id,
            refresh_start,
            refresh_end,
        )

        scored_repo = ParquetScoredNewsRepository(output_dir=paths["processed_news_scored"])
        sentiment_model = FinBERTSentimentModel()
        InferSentimentUseCase(
            news_repository=news_repo,
            sentiment_model=sentiment_model,
            scored_news_repository=scored_repo,
        ).execute(
            asset_id,
            refresh_start,
            refresh_end,
        )

        daily_sentiment_repo = ParquetDailySentimentRepository(
            output_dir=paths["processed_sentiment_daily"]
        )
        SentimentFeatureEngineeringUseCase(
            scored_news_repository=scored_repo,
            sentiment_aggregator=SentimentAggregator(),
            daily_sentiment_repository=daily_sentiment_repo,
        ).execute(
            asset_id,
            refresh_start,
            refresh_end,
        )

        indicator_repo = ParquetTechnicalIndicatorRepository(
            output_dir=paths["processed_technical_indicators"] / asset_id,
            overwrite=True,
        )
        TechnicalIndicatorEngineeringUseCase(
            candle_repository=candles_repo,
            indicator_calculator=TechnicalIndicatorCalculator(),
            indicator_repository=indicator_repo,
        ).execute(asset_id)

        fundamental_repo = ParquetFundamentalRepository(output_dir=paths["processed_fundamentals"])
        fundamental_fetcher = AlphaVantageFundamentalFetcher(api_key=api_key)
        FetchFundamentalsUseCase(
            fundamental_fetcher=fundamental_fetcher,
            fundamental_repository=fundamental_repo,
        ).execute(
            asset_id=asset_id,
            start_date=refresh_start,
            end_date=refresh_end,
        )

        dataset_repo = ParquetTFTDatasetRepository(output_dir=paths["dataset_tft"])
        BuildTFTDatasetUseCase(
            candle_repository=candles_repo,
            indicator_repository=indicator_repo,
            daily_sentiment_repository=daily_sentiment_repo,
            fundamental_repository=fundamental_repo,
            tft_dataset_repository=dataset_repo,
        ).execute(
            asset_id=asset_id,
            start_date=rebuild_start,
            end_date=refresh_end,
        )

    return _refresh_dataset_incremental


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    file_config = _load_json_config(args.config_json)

    asset = (args.asset or file_config.get("asset") or "").strip().upper()
    model_path = (args.model_path or file_config.get("model_path") or "").strip()
    if not asset:
        raise ValueError("Missing required parameter: --asset or config field `asset`.")
    if not model_path:
        raise ValueError(
            "Missing required parameter: --model-path or config field `model_path`."
        )

    start = _parse_yyyymmdd(args.start or file_config.get("start"))
    end = _parse_yyyymmdd(args.end or file_config.get("end"))
    overwrite = bool(file_config.get("overwrite", False)) or bool(args.overwrite)
    overwrite_on_collision = (
        bool(file_config.get("overwrite_on_collision", False))
        or bool(getattr(args, "overwrite_on_collision", False))
    )
    auto_refresh = bool(file_config.get("auto_refresh", False)) or bool(args.auto_refresh)
    strict_quantiles = bool(file_config.get("strict_quantiles", True)) and not bool(
        args.allow_missing_quantiles
    )
    inference_mode = (
        str(args.inference_mode)
        if args.inference_mode is not None
        else str(file_config.get("inference_mode", "rolling"))
    ).strip().lower()
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(file_config.get("batch_size", 64))
    )

    paths = load_data_paths()
    model_path = _resolve_model_path_input(model_path, asset, paths["models"])
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else paths["dataset_tft"]
    inference_dir = Path(args.inference_dir) if args.inference_dir else paths["inference_tft"]

    dataset_repo = ParquetTFTDatasetRepository(output_dir=dataset_dir)
    inference_repo = ParquetTFTInferenceRepository(output_dir=inference_dir)
    analytics_repo = ParquetAnalyticsRunRepository(output_dir=paths["analytics_silver"])
    model_loader = LocalTFTInferenceModelLoader()
    engine = PytorchForecastingTFTInferenceEngine()
    refresh_fn = _build_refresh_fn(paths) if auto_refresh else None

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=model_loader,
        inference_engine=engine,
        analytics_run_repository=analytics_repo,
        refresh_dataset_fn=refresh_fn,
    )

    result = use_case.execute(
        asset_id=asset,
        model_path=model_path,
        start_date=start,
        end_date=end,
        overwrite=overwrite,
        batch_size=batch_size,
        default_end_date=ensure_utc(datetime.now()),
        strict_quantiles=strict_quantiles,
        inference_mode=inference_mode,
        overwrite_on_collision=overwrite_on_collision,
    )

    logger.info(
        "TFT inference completed",
        extra={
            "asset_id": result.asset_id,
            "model_version": result.model_version,
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "inferred": result.inferred,
            "skipped_existing": result.skipped_existing,
            "attempted_upserts": result.attempted_upserts,
            "refreshed_dataset": result.refreshed_dataset,
            "auto_refresh": auto_refresh,
            "strict_quantiles": strict_quantiles,
            "inference_mode": inference_mode,
            "inference_dir": str(inference_dir.resolve()),
        },
    )


if __name__ == "__main__":
    main()
