"""Typed job records — the provenance/tracking contract for a run."""

from __future__ import annotations

import glob
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobState(str, Enum):
    pending = "pending"  # prepared, not yet submitted
    submitted = "submitted"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL: frozenset[JobState] = frozenset(
    {JobState.completed, JobState.failed, JobState.cancelled}
)


class JobRecord(BaseModel):
    """Everything needed to track, resume, and reproduce one tool run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    tool: str
    version: str = "unknown"
    params: dict[str, Any] = Field(default_factory=dict)
    out_dir: str
    script_path: str
    output_globs: dict[str, str] = Field(default_factory=dict)
    job_id: str | None = None
    state: JobState = JobState.pending
    exit_code: int | None = None
    outputs: dict[str, list[str]] = Field(default_factory=dict)
    created_at: str
    submitted_at: str | None = None
    finished_at: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.state == JobState.completed


class RunResult(BaseModel):
    record: JobRecord
    success: bool


def collect_outputs(output_globs: dict[str, str], out_dir: str) -> dict[str, list[str]]:
    """Resolve each named output glob (relative to out_dir) to matched files."""
    resolved: dict[str, list[str]] = {}
    for name, pattern in output_globs.items():
        resolved[name] = sorted(glob.glob(pattern.replace("{out_dir}", out_dir)))
    return resolved
