from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CLITFTTrainRunner:
    @staticmethod
    def _key_to_flag(key: str) -> str:
        return "--" + key.replace("_", "-")

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _pick_new_version(models_asset_dir: Path, before_versions: set[str]) -> str | None:
        after_versions = {p.name for p in models_asset_dir.iterdir() if p.is_dir()}
        created = sorted(after_versions - before_versions)
        if not created:
            return None
        return max(created, key=lambda v: (models_asset_dir / v).stat().st_mtime)

    def run(
        self,
        *,
        asset: str,
        features: str | None,
        config: dict[str, Any],
        split_config: dict[str, str] | None,
        models_asset_dir: Path,
    ) -> tuple[str | None, dict[str, Any] | None]:
        models_asset_dir.mkdir(parents=True, exist_ok=True)
        before_versions = {p.name for p in models_asset_dir.iterdir() if p.is_dir()}
        models_base_dir = models_asset_dir.parent
        cmd = [sys.executable, "-m", "src.main_train_tft", "--asset", asset]
        cmd.extend(["--models-dir", str(models_base_dir)])
        if features:
            cmd.extend(["--features", features])
        if split_config:
            split_flag_map = {
                "train_start": "--train-start",
                "train_end": "--train-end",
                "val_start": "--val-start",
                "val_end": "--val-end",
                "test_start": "--test-start",
                "test_end": "--test-end",
            }
            for key, flag in split_flag_map.items():
                value = split_config.get(key)
                if value is not None:
                    cmd.extend([flag, str(value)])
        for key, value in config.items():
            cmd.extend([self._key_to_flag(key), str(value)])

        logger.info("Starting analysis run", extra={"cmd": cmd})
        subprocess.run(cmd, check=True)

        version = self._pick_new_version(models_asset_dir, before_versions)
        if version is None:
            return None, None
        metadata_path = models_asset_dir / version / "metadata.json"
        if not metadata_path.exists():
            return version, None
        return version, self._load_metadata(metadata_path)
