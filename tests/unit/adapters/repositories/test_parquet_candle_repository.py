# tests/unit/adapters/repositories/test_parquet_candle_repository.py

import pytest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.entities.candle import Candle
from src.infrastructure.schemas.candle_parquet_schema import (
    CANDLE_PARQUET_COLUMNS,
)


# Helpers / Fixtures

@pytest.fixture
def candles_sample() -> list[Candle]:
    base_time = datetime(2024, 1, 1)
    return [
        Candle(
            timestamp=base_time + timedelta(days=i),
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100.5 + i,
            volume=1000 + i,
        )
        for i in range(3)
    ]


@pytest.fixture
def repo(tmp_path: Path) -> ParquetCandleRepository:
    return ParquetCandleRepository(output_dir=tmp_path)


# Constructor validation

def test_repository_raises_if_directory_does_not_exist(tmp_path: Path):
    invalid_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        ParquetCandleRepository(output_dir=invalid_dir)


def test_repository_raises_if_path_is_not_directory(tmp_path: Path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir")

    with pytest.raises(NotADirectoryError):
        ParquetCandleRepository(output_dir=file_path)


# Save behavior

def test_save_candles_persists_parquet_file(repo, candles_sample, tmp_path):
    repo.save_candles("AAPL", candles_sample)

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    assert "AAPL" in files[0].name


def test_save_candles_normalizes_symbol(repo, candles_sample, tmp_path):
    repo.save_candles("PETR4.SA", candles_sample)

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    assert files[0].name == "candles_PETR4_1d.parquet"


def test_save_candles_raises_on_empty_input(repo):
    with pytest.raises(ValueError):
        repo.save_candles("AAPL", [])


# Load behavior

def test_load_candles_roundtrip_preserves_data(repo, candles_sample):
    repo.save_candles("AAPL", candles_sample)

    loaded = repo.load_candles("AAPL")

    assert len(loaded) == len(candles_sample)
    assert all(isinstance(c, Candle) for c in loaded)
    assert loaded[0].timestamp == candles_sample[0].timestamp
    assert loaded[0].close == candles_sample[0].close


def test_load_candles_returns_sorted_by_timestamp(repo, candles_sample):
    shuffled = list(reversed(candles_sample))
    repo.save_candles("AAPL", shuffled)

    loaded = repo.load_candles("AAPL")

    timestamps = [c.timestamp for c in loaded]
    assert timestamps == sorted(timestamps)


def test_load_raises_if_file_not_found(repo):
    with pytest.raises(FileNotFoundError):
        repo.load_candles("MSFT")


# Schema validation

def test_load_raises_if_schema_is_invalid(repo, tmp_path, candles_sample):
    # Create an invalid parquet manually
    df = pd.DataFrame(
        [
            {
                "timestamp": candles_sample[0].timestamp,
                "open": 10.0,
                # missing required columns
            }
        ]
    )

    filepath = tmp_path / "candles_AAPL_1d.parquet"
    df.to_parquet(filepath, index=False)

    with pytest.raises(ValueError):
        repo.load_candles("AAPL")


def test_saved_parquet_has_expected_columns(repo, candles_sample, tmp_path):
    repo.save_candles("AAPL", candles_sample)

    filepath = tmp_path / "candles_AAPL_1d.parquet"
    df = pd.read_parquet(filepath)

    assert set(df.columns) == CANDLE_PARQUET_COLUMNS


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(architecture):
# Expor política explícita de persistência:
# overwrite | append | upsert
# permitindo evolução do pipeline sem modificar o adapter.

# TODO(stat-validation):
# Validar gaps temporais excessivos na série de candles
# (ex: dias úteis ausentes), sinalizando possíveis falhas de coleta.

# TODO(test-quality):
# Adicionar testes de roundtrip (save → load)
# garantindo preservação de ordenação temporal e valores numéricos.

# TODO(reproducibility):
# Persistir metadata do dataset gerado
# (ex: período, fonte, hash do arquivo Parquet).