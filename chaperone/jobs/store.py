"""Run store — one JSON file per run for tracking, listing, and resuming."""

from __future__ import annotations

from pathlib import Path

from chaperone.jobs.records import JobRecord


class RunStore:
    def __init__(self, runs_dir: Path | str) -> None:
        self._dir = Path(runs_dir)

    def save(self, record: JobRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{record.run_id}.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )

    def load(self, run_id: str) -> JobRecord:
        return JobRecord.model_validate_json(
            (self._dir / f"{run_id}.json").read_text(encoding="utf-8")
        )

    def list(self) -> list[JobRecord]:
        if not self._dir.is_dir():
            return []
        records = [
            JobRecord.model_validate_json(p.read_text(encoding="utf-8"))
            for p in self._dir.glob("*.json")
        ]
        return sorted(records, key=lambda r: r.created_at, reverse=True)
