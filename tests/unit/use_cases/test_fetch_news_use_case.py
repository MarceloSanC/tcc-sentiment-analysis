# tests/unit/use_cases/test_fetch_news_use_case.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pytest

from src.entities.news_article import NewsArticle
from src.use_cases.fetch_news_use_case import FetchNewsUseCase


def _dt_utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def _article(
    asset_id: str,
    published_at: datetime,
    url: str,
    *,
    headline: str = "h",
    summary: str = "s",
    source: str = "alpha_vantage",
) -> NewsArticle:
    # NewsArticle invariants: tz-aware published_at + source non-empty + url http(s)
    return NewsArticle(
        asset_id=asset_id,
        published_at=published_at,
        headline=headline,
        summary=summary,
        source=source,
        url=url,
        article_id=url,  # policy: id = url
        language="en",
    )


class FakeNewsFetcher:
    """
    Scripted fetcher: returns a predefined batch for each call.

    Keyed by (start_iso, end_iso) to let us validate cursor behavior.
    """

    def __init__(self, scripted: Dict[Tuple[str, str], List[NewsArticle]]) -> None:
        self.scripted = scripted
        self.calls: list[tuple[str, datetime, datetime]] = []

    def fetch_company_news(self, asset_id: str, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        self.calls.append((asset_id, start_date, end_date))
        key = (start_date.isoformat(), end_date.isoformat())
        return list(self.scripted.get(key, []))


@dataclass
class UpsertCall:
    asset_id: str
    urls: list[str]
    published_ats: list[datetime]


class FakeNewsRepository:
    """
    In-memory repo that simulates:
    - get_latest_published_at
    - list_news (for seed window)
    - upsert_news_batch recording
    """

    def __init__(
        self,
        *,
        latest: datetime | None = None,
        seed_window_rows: list[NewsArticle] | None = None,
    ) -> None:
        self._latest = latest
        self._seed_window_rows = seed_window_rows or []

        self.upsert_calls: list[UpsertCall] = []
        self.persisted_by_url: dict[str, NewsArticle] = {}

    def get_latest_published_at(self, asset_id: str) -> datetime | None:
        return self._latest

    def list_news(self, asset_id: str, start_date: datetime, end_date: datetime) -> list[NewsArticle]:
        # Return whatever the test configured as "already persisted near cursor"
        return list(self._seed_window_rows)

    def upsert_news_batch(self, articles: list[NewsArticle]) -> None:
        if not articles:
            raise ValueError("test harness expects no empty upsert")

        asset = articles[0].asset_id
        assert all(a.asset_id == asset for a in articles), "batch must be single asset"

        urls = [a.url for a in articles if a.url is not None]
        self.upsert_calls.append(
            UpsertCall(
                asset_id=asset,
                urls=urls,
                published_ats=[a.published_at for a in articles],
            )
        )

        # emulate upsert by URL/article_id: last occurrence wins
        for a in articles:
            assert a.url is not None, "repo policy expects url for stable id"
            self.persisted_by_url[a.url] = a

        # update latest persisted to max(published_at)
        max_dt = max(a.published_at for a in articles)
        if self._latest is None or max_dt > self._latest:
            self._latest = max_dt


def test_fetch_news_dedups_overlap_by_url_and_advances_cursor_correctly():
    """
    Main acceptance for dataset-building behavior:
    - next request starts exactly at latest_dt of prior batch (overlap)
    - duplicates from overlap are removed by URL (global seen_urls)
    - repository receives only "new" articles
    - last_cursor advances to the batch latest_dt
    """
    asset = "AAPL"
    start = _dt_utc(2010, 1, 1, 0, 0)
    end = _dt_utc(2010, 1, 1, 0, 3)

    # batch1 returns 3 items; latest_dt = 00:01
    a1 = _article(asset, _dt_utc(2010, 1, 1, 0, 0), "https://n/1")
    a2 = _article(asset, _dt_utc(2010, 1, 1, 0, 1), "https://n/2")
    a3 = _article(asset, _dt_utc(2010, 1, 1, 0, 1), "https://n/3")

    # batch2 starts at latest_dt (00:01) and contains:
    # - duplicates of a2,a3 (same URLs)
    # - plus new ones after that
    b2_dup_2 = _article(asset, _dt_utc(2010, 1, 1, 0, 1), "https://n/2")
    b2_dup_3 = _article(asset, _dt_utc(2010, 1, 1, 0, 1), "https://n/3")
    b2_new_4 = _article(asset, _dt_utc(2010, 1, 1, 0, 2), "https://n/4")
    b2_new_5 = _article(asset, _dt_utc(2010, 1, 1, 0, 3), "https://n/5")

    # Script per (cursor,end) in isoformat
    scripted = {
        (start.isoformat(), end.isoformat()): [a1, a2, a3],
        (_dt_utc(2010, 1, 1, 0, 1).isoformat(), end.isoformat()): [b2_dup_2, b2_dup_3, b2_new_4, b2_new_5],
        # third call would be after latest_dt=00:03; we can omit because we expect stop at near_end
    }

    fetcher = FakeNewsFetcher(scripted=scripted)
    repo = FakeNewsRepository(latest=None, seed_window_rows=[])

    use_case = FetchNewsUseCase(
        news_fetcher=fetcher,
        news_repository=repo,
        safety_margin=950,                    # small batches will be < margin
        cursor_step=timedelta(minutes=1),     # near_end check
        empty_batch_advance=timedelta(days=1),
    )

    result = use_case.execute(asset_id=asset, start_date=start, end_date=end)

    # Fetch calls: first at start, then at latest_dt of batch1 (00:01)
    assert len(fetcher.calls) >= 2
    assert fetcher.calls[0][1] == start
    assert fetcher.calls[1][1] == _dt_utc(2010, 1, 1, 0, 1)

    # Upserts:
    # - first upsert saves 3
    # - second upsert saves only the 2 new ones (n/4, n/5)
    assert len(repo.upsert_calls) == 2
    assert repo.upsert_calls[0].urls == ["https://n/1", "https://n/2", "https://n/3"]
    assert repo.upsert_calls[1].urls == ["https://n/4", "https://n/5"]

    # Result counters
    assert result.asset_id == asset
    assert result.saved == 5
    assert result.fetched == 3 + 4  # fetched includes duplicates returned by API
    assert result.last_cursor == _dt_utc(2010, 1, 1, 0, 3)


def test_fetch_news_resumes_from_latest_persisted_and_seeds_seen_urls():
    """
    Incremental update behavior:
    - If repo already has latest_published_at, cursor starts there (or start_date, whichever is later)
    - Seed window reads existing news near cursor and prevents re-upserting them (URL dedup)
    """
    asset = "AAPL"
    start = _dt_utc(2010, 1, 1, 0, 0)
    end = _dt_utc(2010, 1, 1, 0, 5)

    latest_persisted = _dt_utc(2010, 1, 1, 0, 2)  # resume cursor at 00:02
    already = _article(asset, _dt_utc(2010, 1, 1, 0, 2), "https://n/seeded")

    # First fetch at cursor=latest_persisted returns seeded duplicate + new
    dup = _article(asset, _dt_utc(2010, 1, 1, 0, 2), "https://n/seeded")
    new1 = _article(asset, _dt_utc(2010, 1, 1, 0, 3), "https://n/new1")
    new2 = _article(asset, _dt_utc(2010, 1, 1, 0, 5), "https://n/new2")

    scripted = {
        (latest_persisted.isoformat(), end.isoformat()): [dup, new1, new2],
    }

    fetcher = FakeNewsFetcher(scripted=scripted)
    repo = FakeNewsRepository(latest=latest_persisted, seed_window_rows=[already])

    use_case = FetchNewsUseCase(
        news_fetcher=fetcher,
        news_repository=repo,
        safety_margin=950,
        cursor_step=timedelta(minutes=1),
    )

    result = use_case.execute(asset_id=asset, start_date=start, end_date=end)

    # Cursor starts at latest persisted (00:02)
    assert fetcher.calls[0][1] == latest_persisted

    # Upsert should NOT contain the seeded url
    assert len(repo.upsert_calls) == 1
    assert repo.upsert_calls[0].urls == ["https://n/new1", "https://n/new2"]

    assert result.saved == 2
    assert result.fetched == 3
    assert result.last_cursor == _dt_utc(2010, 1, 1, 0, 5)


def test_fetch_news_empty_batch_advances_cursor_by_configured_delta():
    """
    Sparse periods happen (no news). The use case must advance cursor (avoids infinite loops).
    """
    asset = "AAPL"
    start = _dt_utc(2010, 1, 1, 0, 0)
    end = _dt_utc(2010, 1, 2, 0, 0)

    # First call returns empty, so cursor advances by empty_batch_advance (1 day).
    # Second call at +1 day returns 1 item near end, allowing stop.
    cursor2 = start + timedelta(days=1)
    a = _article(asset, _dt_utc(2010, 1, 2, 0, 0), "https://n/1")

    scripted = {
        (start.isoformat(), end.isoformat()): [],
        (cursor2.isoformat(), end.isoformat()): [a],
    }

    fetcher = FakeNewsFetcher(scripted=scripted)
    repo = FakeNewsRepository()

    use_case = FetchNewsUseCase(
        news_fetcher=fetcher,
        news_repository=repo,
        safety_margin=950,
        empty_batch_advance=timedelta(days=1),
    )

    result = use_case.execute(asset_id=asset, start_date=start, end_date=end)

    assert len(fetcher.calls) >= 2
    assert fetcher.calls[0][1] == start
    assert fetcher.calls[1][1] == cursor2

    assert result.saved == 1
    assert repo.upsert_calls[0].urls == ["https://n/1"]


def test_fetch_news_stalling_protection_forces_step_forward_when_no_progress():
    """
    If the API returns only duplicates at the same timestamp, cursor doesn't advance and
    no new rows are saved -> must force cursor forward by cursor_step to avoid infinite loop.
    """
    asset = "AAPL"
    start = _dt_utc(2010, 1, 1, 0, 0)
    end = _dt_utc(2010, 1, 1, 0, 10)

    # Repo has the URL already (seeded), so fetch returns only duplicates at same timestamp.
    dup = _article(asset, _dt_utc(2010, 1, 1, 0, 0), "https://n/dup")

    # First call at start returns only dup (which will be filtered out by seen_urls),
    # so no save and cursor_next == cursor -> stalling => forced step to +1 minute.
    scripted = {
        (start.isoformat(), end.isoformat()): [dup],
        ((start + timedelta(minutes=1)).isoformat(), end.isoformat()): [],  # then empty and will advance
    }

    fetcher = FakeNewsFetcher(scripted=scripted)
    repo = FakeNewsRepository(latest=None, seed_window_rows=[dup])

    use_case = FetchNewsUseCase(
        news_fetcher=fetcher,
        news_repository=repo,
        safety_margin=950,
        cursor_step=timedelta(minutes=1),
        empty_batch_advance=timedelta(days=1),
        max_iterations_per_asset=5,
    )

    result = use_case.execute(asset_id=asset, start_date=start, end_date=end)

    # First call at start, second call should be forced to +1 minute due to stalling
    assert fetcher.calls[0][1] == start
    assert fetcher.calls[1][1] == start + timedelta(minutes=1)

    # No upserts because everything was duplicate and then empty
    assert repo.upsert_calls == []
    assert result.saved == 0


def test_fetch_news_rejects_naive_start_or_end():
    asset = "AAPL"
    use_case = FetchNewsUseCase(
        news_fetcher=FakeNewsFetcher(scripted={}),
        news_repository=FakeNewsRepository(),
    )

    with pytest.raises(ValueError):
        use_case.execute(
            asset_id=asset,
            start_date=datetime(2010, 1, 1),  # naive
            end_date=_dt_utc(2010, 1, 2),
        )

    with pytest.raises(ValueError):
        use_case.execute(
            asset_id=asset,
            start_date=_dt_utc(2010, 1, 1),
            end_date=datetime(2010, 1, 2),  # naive
        )
