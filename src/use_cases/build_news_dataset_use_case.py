# src/use_cases/build_news_dataset_use_case.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.news_article import NewsArticle
from src.interfaces.news_fetcher import NewsFetcher
from src.interfaces.news_repository import NewsRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildNewsDatasetResult:
    asset_id: str
    fetched: int
    saved: int
    iterations: int
    start: datetime
    end: datetime
    last_cursor: datetime


class BuildNewsDatasetUseCase:
    """
    Build (and incrementally update) a historical news dataset by walking forward in time.

    Core strategy (Alpha Vantage constraints):
    - Query with sort=EARLIEST, limit=1000.
    - Move cursor forward using the max(published_at) from the last batch.
    - Next request starts exactly at last batch latest_dt (same-minute overlap),
      so we must deduplicate to avoid duplicates and to avoid missing overflow at boundaries.

    Dedup policy:
    - Primary: URL (stable in Alpha Vantage). We keep a global 'seen_urls' set during this run.
    - Repository upsert still dedups by article_id (which we set to URL).

    Stalling protection:
    - If cursor does not advance AND we saved nothing new -> bump cursor by cursor_step.
    """

    def __init__(
        self,
        news_fetcher: NewsFetcher,
        news_repository: NewsRepository,
        safety_margin: int = 950,
        max_iterations_per_asset: int = 20000,
        cursor_step: timedelta = timedelta(minutes=1),
        seed_back_window: timedelta = timedelta(days=2),
        seed_forward_window: timedelta = timedelta(days=1),
        empty_batch_advance: timedelta = timedelta(days=1),
    ) -> None:
        self.news_fetcher = news_fetcher
        self.news_repository = news_repository
        self.safety_margin = int(safety_margin)
        self.max_iterations_per_asset = int(max_iterations_per_asset)
        self.cursor_step = cursor_step
        self.seed_back_window = seed_back_window
        self.seed_forward_window = seed_forward_window
        self.empty_batch_advance = empty_batch_advance

    @staticmethod
    def _dedup_by_url_in_batch(articles: Iterable[NewsArticle]) -> list[NewsArticle]:
        """
        Keep first occurrence of each URL within the batch.
        Articles without URL are kept (but will likely be rejected later by repository policy).
        """
        seen: set[str] = set()
        out: list[NewsArticle] = []
        for a in articles:
            if a.url:
                if a.url in seen:
                    continue
                seen.add(a.url)
            out.append(a)
        return out

    @staticmethod
    def _extract_urls(articles: Iterable[NewsArticle]) -> set[str]:
        urls: set[str] = set()
        for a in articles:
            if a.url:
                urls.add(a.url)
        return urls

    def execute(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        *,
        seed_seen_urls: Optional[set[str]] = None,
    ) -> BuildNewsDatasetResult:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        # Incremental cursor: if we already have data, resume from latest persisted
        latest_persisted = self.news_repository.get_latest_published_at(asset_id)
        if latest_persisted is not None:
            require_tz_aware(latest_persisted, "latest_persisted")
            latest_persisted_utc = to_utc(latest_persisted)
            cursor = max(start_utc, latest_persisted_utc)
        else:
            cursor = start_utc

        iterations = 0
        fetched_total = 0
        saved_total = 0

        # Global dedup by URL for this run
        seen_urls: set[str] = set(seed_seen_urls or set())

        # Seed dedup around cursor (cheap overlap protection for reruns)
        try:
            seed_start = max(start_utc, cursor - self.seed_back_window)
            seed_end = min(end_utc, cursor + self.seed_forward_window)
            seed = self.news_repository.list_news(asset_id, seed_start, seed_end)
            seen_urls |= self._extract_urls(seed)
            logger.info(
                "Seeded seen_urls from repository window",
                extra={
                    "asset": asset_id,
                    "seed_start": seed_start.isoformat(),
                    "seed_end": seed_end.isoformat(),
                    "seeded_urls": len(seen_urls),
                },
            )
        except Exception as e:
            # Se seed falhar por qualquer motivo, não bloqueia o build (apenas loga)
            logger.warning(
                "Failed to seed seen_urls from repository; continuing",
                extra={"asset": asset_id, "error": str(e)},
            )

        stall_count = 0

        logger.info(
            "Building news dataset (start)",
            extra={
                "asset": asset_id,
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
                "cursor": cursor.isoformat(),
                "safety_margin": self.safety_margin,
            },
        )

        while cursor <= end_utc and iterations < self.max_iterations_per_asset:
            iterations += 1

            batch = self.news_fetcher.fetch_company_news(asset_id, cursor, end_utc)
            batch = self._dedup_by_url_in_batch(batch)
            fetched_total += len(batch)

            if not batch:
                cursor_next = cursor + self.empty_batch_advance
                logger.info(
                    "Empty batch; advancing cursor",
                    extra={
                        "asset": asset_id,
                        "cursor": cursor.isoformat(),
                        "cursor_next": cursor_next.isoformat(),
                    },
                )
                cursor = cursor_next
                stall_count = 0
                continue

            # Latest dt in batch (must be UTC-aware)
            latest_dt = max(to_utc(a.published_at) for a in batch)

            # Filter out global duplicates by URL
            new_articles: list[NewsArticle] = []
            for a in batch:
                if a.url and a.url in seen_urls:
                    continue
                if a.url:
                    seen_urls.add(a.url)
                new_articles.append(a)

            if new_articles:
                # Repository does upsert by article_id (url), but we already pre-dedup by URL
                self.news_repository.upsert_news_batch(new_articles)
                saved_total += len(new_articles)

            # logger.info(
            #     "News dataset step",
            #     extra={
            #         "asset": asset_id,
            #         "cursor": cursor.isoformat(),
            #         "latest_dt": latest_dt.isoformat(),
            #         "fetched_batch": len(batch),
            #         "saved_batch": len(new_articles),
            #         "iterations": iterations,
            #     },
            # )

            # Stop condition:
            # - If batch is below safety margin AND we are already at/near the end, assume no more pages.
            near_end = latest_dt >= (end_utc - self.cursor_step)
            if len(batch) < self.safety_margin and near_end:
                cursor = latest_dt
                break

            # Cursor advancement rule:
            # next cursor := latest_dt (same-minute overlap handled by dedup)
            cursor_next = latest_dt

            # Stalling protection:
            if cursor_next <= cursor and len(new_articles) == 0:
                stall_count += 1
                forced_next = cursor + self.cursor_step
                logger.warning(
                    "Cursor stalled at %s; forcing step forward for %s",
                    cursor.isoformat(),
                    forced_next.isoformat(),
                    extra={
                        "asset": asset_id,
                        "cursor": cursor.isoformat(),
                        "forced_next": forced_next.isoformat(),
                        "stall_count": stall_count,
                    },
                )
                cursor = forced_next
            else:
                stall_count = 0
                cursor = cursor_next

        if iterations >= self.max_iterations_per_asset:
            logger.warning(
                "Max iterations reached; dataset build may be incomplete",
                extra={
                    "asset": asset_id,
                    "iterations": iterations,
                    "cursor": cursor.isoformat(),
                },
            )

        logger.info(
            "Building news dataset (done)",
            extra={
                "asset": asset_id,
                "fetched_total": fetched_total,
                "saved_total": saved_total,
                "iterations": iterations,
                "last_cursor": cursor.isoformat(),
            },
        )

        return BuildNewsDatasetResult(
            asset_id=asset_id,
            fetched=fetched_total,
            saved=saved_total,
            iterations=iterations,
            start=start_utc,
            end=end_utc,
            last_cursor=cursor,
        )
