"""Job lifecycle: prepare → submit → monitor → collect, with provenance persisted.

Each run gets its own directory under ``paths.runs_dir`` holding the rendered script,
the scheduler log, the tool's outputs, and a JSON record updated at every state change.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from chaperone.jobs.records import TERMINAL, JobRecord, JobState, RunResult, collect_outputs
from chaperone.jobs.scheduler import Scheduler
from chaperone.jobs.store import RunStore
from chaperone.settings import Settings
from chaperone.skills.manifest import ToolManifest
from chaperone.skills.render import RenderedJob, render_job
from chaperone.utils.logger import logger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRunner:
    def __init__(self, settings: Settings, scheduler: Scheduler) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._store = RunStore(settings.paths.runs_dir)

    def render(self, manifest: ToolManifest, params: dict, *, run_id: str | None = None) -> RenderedJob:
        """Validate + render only (no filesystem writes, no submission) — the dry-run path."""
        run_id = run_id or manifest.name
        out_dir = str(Path(self._settings.paths.runs_dir) / run_id / "out")
        return render_job(manifest, params, out_dir=out_dir, job_name=run_id)

    def prepare(self, manifest: ToolManifest, params: dict) -> JobRecord:
        run_id = f"{manifest.name}-{uuid.uuid4().hex[:8]}"
        run_dir = Path(self._settings.paths.runs_dir) / run_id
        out_dir = run_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        rendered = render_job(manifest, params, out_dir=str(out_dir), job_name=run_id)
        script_path = run_dir / "job.sh"
        script_path.write_text(rendered.script, encoding="utf-8")

        record = JobRecord(
            run_id=run_id,
            tool=manifest.name,
            version=manifest.version,
            params=rendered.params,
            out_dir=str(out_dir),
            script_path=str(script_path),
            output_globs=rendered.output_globs,
            created_at=_now(),
        )
        self._store.save(record)
        return record

    def submit(self, record: JobRecord) -> JobRecord:
        record.job_id = self._scheduler.submit(record.script_path, record.out_dir)
        record.state = JobState.submitted
        record.submitted_at = _now()
        self._store.save(record)
        logger.info(f"Submitted {record.tool} run {record.run_id} (job {record.job_id}).")
        return record

    def wait(self, record: JobRecord, *, timeout: float | None = None) -> RunResult:
        start = time.monotonic()
        while record.state not in TERMINAL:
            state = self._scheduler.poll(record.job_id or "")
            if state != record.state:
                record.state = state
                self._store.save(record)
            if state in TERMINAL:
                break
            if timeout is not None and time.monotonic() - start > timeout:
                logger.warning(f"Run {record.run_id} still {state} after {timeout}s; not waiting longer.")
                break
            time.sleep(self._settings.jobs.poll_interval)

        record.finished_at = _now()
        record.exit_code = self._scheduler.exit_code(record.job_id or "")
        record.outputs = collect_outputs(record.output_globs, record.out_dir)
        self._store.save(record)
        return RunResult(record=record, success=record.succeeded)

    def run(self, manifest: ToolManifest, params: dict, *, watch: bool = True) -> RunResult:
        record = self.submit(self.prepare(manifest, params))
        if watch:
            return self.wait(record)
        return RunResult(record=record, success=False)  # submitted; outcome not yet known
