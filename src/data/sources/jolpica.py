"""HTTP client for the Jolpica F1 API, the community-maintained successor to
the now-defunct Ergast API (https://api.jolpi.ca/ergast/).

Jolpica is schema-compatible with Ergast, which is the source RelBench itself
used to build ``rel-f1`` (see ``relbench.datasets.f1.F1Dataset``). We rely on
this compatibility to append seasons that the frozen RelBench snapshot does
not cover, without inventing a new schema.

Every response is cached twice:
  1. via ``requests_cache`` (HTTP-level, sqlite-backed) so re-running the
     pipeline never re-hits the network for endpoints already seen;
  2. as raw indented JSON snapshots under ``raw_dir``, for auditability and
     so the enrichment pipeline can be rebuilt fully offline from the
     snapshots alone.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests_cache

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_LIMIT = 100
MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 1.5
REQUEST_TIMEOUT = 20
POLITE_DELAY_SECONDS = 1.0  # be a good API citizen between uncached calls


class JolpicaClient:
    """Thin, resilient wrapper around the Jolpica (Ergast-successor) REST API."""

    def __init__(
        self,
        raw_dir: str = "data/raw/jolpica",
        http_cache_path: Optional[str] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        cache_path = http_cache_path or str(self.raw_dir / "http_cache")
        self.session = requests_cache.CachedSession(
            cache_name=cache_path,
            backend="sqlite",
            expire_after=-1,  # never expire automatically; use force_refresh to bust
            allowable_methods=("GET",),
        )

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        params = params or {}

        for attempt in range(1, MAX_RETRIES + 1):
            response = self.session.get(
                url, params=params, timeout=REQUEST_TIMEOUT, force_refresh=force_refresh
            )

            if response.status_code == 200:
                if not getattr(response, "from_cache", False):
                    time.sleep(POLITE_DELAY_SECONDS)
                return response.json()["MRData"]

            if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                delay += random.uniform(0, 0.5)
                logger.warning(
                    "Jolpica %s returned %s (attempt %d/%d), retrying in %.1fs",
                    url, response.status_code, attempt, MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()

        raise RuntimeError(f"Exhausted retries fetching {url} with params={params}")

    def _write_snapshot(self, snapshot_name: str, data: Dict[str, Any]) -> None:
        snapshot_path = self.raw_dir / f"{snapshot_name}.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def get_snapshot(
        self,
        path: str,
        snapshot_name: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a JSON endpoint and persist the raw response for auditability."""
        data = self._get(path, params, force_refresh=force_refresh)
        self._write_snapshot(snapshot_name, data)
        return data

    # ------------------------------------------------------------------
    # Paginated collection endpoints (season schedule, drivers, etc.)
    # ------------------------------------------------------------------

    def get_all_pages(
        self,
        path: str,
        extractor: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        params: Optional[Dict[str, Any]] = None,
        snapshot_name: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        params = dict(params or {})
        items: List[Dict[str, Any]] = []
        offset = 0

        while True:
            page_params = {**params, "limit": limit, "offset": offset}
            data = self._get(path, page_params, force_refresh=force_refresh)
            total = int(data.get("total", 0))
            batch = extractor(data)
            items.extend(batch)

            if snapshot_name:
                self._write_snapshot(f"{snapshot_name}_offset{offset}", data)

            offset += limit
            if offset >= total or not batch:
                break

        return items

    # ------------------------------------------------------------------
    # Domain-specific convenience methods
    # ------------------------------------------------------------------

    def get_season_schedule(self, season: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}.json", f"{season}/schedule", {"limit": DEFAULT_LIMIT}, force_refresh=force_refresh
        )
        return data["RaceTable"]["Races"]

    def get_season_drivers(self, season: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/drivers.json", f"{season}/drivers", {"limit": DEFAULT_LIMIT}, force_refresh=force_refresh
        )
        return data["DriverTable"]["Drivers"]

    def get_season_constructors(self, season: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/constructors.json",
            f"{season}/constructors",
            {"limit": DEFAULT_LIMIT},
            force_refresh=force_refresh,
        )
        return data["ConstructorTable"]["Constructors"]

    def get_circuits(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self.get_all_pages(
            "circuits.json",
            extractor=lambda d: d["CircuitTable"]["Circuits"],
            snapshot_name="circuits",
            force_refresh=force_refresh,
        )

    def get_status_table(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self.get_all_pages(
            "status.json",
            extractor=lambda d: d["StatusTable"]["Status"],
            snapshot_name="status",
            force_refresh=force_refresh,
        )

    def get_race_results(
        self, season: int, round_: int, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/{round_}/results.json",
            f"{season}/{round_}/results",
            {"limit": DEFAULT_LIMIT},
            force_refresh=force_refresh,
        )
        races = data["RaceTable"]["Races"]
        return races[0] if races else None

    def get_qualifying(
        self, season: int, round_: int, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/{round_}/qualifying.json",
            f"{season}/{round_}/qualifying",
            {"limit": DEFAULT_LIMIT},
            force_refresh=force_refresh,
        )
        races = data["RaceTable"]["Races"]
        return races[0] if races else None

    def get_sprint(
        self, season: int, round_: int, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/{round_}/sprint.json",
            f"{season}/{round_}/sprint",
            {"limit": DEFAULT_LIMIT},
            force_refresh=force_refresh,
        )
        races = data["RaceTable"]["Races"]
        return races[0] if races else None

    def get_driver_standings(
        self, season: int, round_: int, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/{round_}/driverStandings.json",
            f"{season}/{round_}/driverStandings",
            {"limit": DEFAULT_LIMIT},
            force_refresh=force_refresh,
        )
        lists = data["StandingsTable"]["StandingsLists"]
        return lists[0] if lists else None

    def get_constructor_standings(
        self, season: int, round_: int, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        data = self.get_snapshot(
            f"{season}/{round_}/constructorStandings.json",
            f"{season}/{round_}/constructorStandings",
            {"limit": DEFAULT_LIMIT},
            force_refresh=force_refresh,
        )
        lists = data["StandingsTable"]["StandingsLists"]
        return lists[0] if lists else None
