"""Scheduler backends.

``SlurmScheduler`` submits and polls real cluster jobs. ``LocalScheduler`` runs the
rendered script as a local subprocess — the same "mock so it works without the real
thing" pattern as the CPU LLM backend, so the whole job lifecycle is exercisable in
tests and demos with no SLURM. (LocalScheduler runs the script verbatim, so it suits
env-light manifests; real tools that `conda activate` / `module load` need SLURM.)
"""

from __future__ import annotations

import os
import subprocess
from typing import Protocol, runtime_checkable

from chaperone.jobs.records import JobState


@runtime_checkable
class Scheduler(Protocol):
    name: str

    def submit(self, script_path: str, out_dir: str) -> str: ...
    def poll(self, job_id: str) -> JobState: ...
    def exit_code(self, job_id: str) -> int | None: ...
    def cancel(self, job_id: str) -> None: ...


class LocalScheduler:
    name = "local"

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}

    def submit(self, script_path: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        log = open(os.path.join(out_dir, "job.log"), "w")
        try:
            proc = subprocess.Popen(["bash", script_path], stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()  # child keeps its own dup of the fd (POSIX)
        self._procs[str(proc.pid)] = proc
        return str(proc.pid)

    def poll(self, job_id: str) -> JobState:
        proc = self._procs.get(job_id)
        if proc is None:
            return JobState.failed
        code = proc.poll()
        if code is None:
            return JobState.running
        return JobState.completed if code == 0 else JobState.failed

    def exit_code(self, job_id: str) -> int | None:
        proc = self._procs.get(job_id)
        return None if proc is None else proc.returncode

    def cancel(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()


class SlurmScheduler:
    name = "slurm"

    def submit(self, script_path: str, out_dir: str) -> str:
        result = subprocess.run(
            ["sbatch", "--parsable", script_path], capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split(";")[0]

    def poll(self, job_id: str) -> JobState:
        # squeue knows active jobs; once gone, sacct has the terminal state.
        active = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True
        ).stdout.strip()
        if active:
            return _map_state(active)
        done = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "-o", "State"], capture_output=True, text=True
        ).stdout.strip()
        return _map_state(done.split()[0]) if done else JobState.completed

    def exit_code(self, job_id: str) -> int | None:
        out = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "-o", "ExitCode"], capture_output=True, text=True
        ).stdout.strip()
        if not out:
            return None
        try:
            return int(out.split()[0].split(":")[0])
        except (ValueError, IndexError):
            return None

    def cancel(self, job_id: str) -> None:
        subprocess.run(["scancel", job_id], check=False)


def _map_state(raw: str) -> JobState:
    s = raw.upper()
    if s in {"PENDING", "CONFIGURING"}:
        return JobState.submitted
    if s in {"RUNNING", "COMPLETING"}:
        return JobState.running
    if s.startswith("CANCELLED"):
        return JobState.cancelled
    if s == "COMPLETED":
        return JobState.completed
    return JobState.failed  # FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, ...
