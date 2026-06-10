"""Render a manifest + params into a ready-to-submit SLURM job script."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from chaperone.skills.manifest import ToolManifest


class RenderedJob(BaseModel):
    tool: str
    params: dict[str, Any]
    out_dir: str
    script: str
    output_globs: dict[str, str]


def render_job(
    manifest: ToolManifest,
    params: dict[str, Any],
    *,
    out_dir: str,
    job_name: str | None = None,
) -> RenderedJob:
    """Validate params and produce the full SLURM script (no side effects)."""
    resolved = manifest.coerce_params(params)
    context = {**resolved, "out_dir": out_dir}
    command = _fill(manifest.command, context)
    script = _build_script(manifest, command, out_dir, job_name or manifest.name)
    return RenderedJob(
        tool=manifest.name,
        params=resolved,
        out_dir=out_dir,
        script=script,
        output_globs=manifest.outputs,
    )


def _fill(template: str, context: dict[str, Any]) -> str:
    """Replace {key} placeholders, leaving bash ``${VAR}`` / brace-expansion alone.

    We substitute only known keys (not ``str.format``) so the command template can
    contain literal shell braces without breaking.
    """
    out = template
    for key, value in context.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _build_script(manifest: ToolManifest, command: str, out_dir: str, job_name: str) -> str:
    r = manifest.resources
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={r.partition}",
    ]
    if r.gres:
        lines.append(f"#SBATCH --gres={r.gres}")
    lines += [
        f"#SBATCH --cpus-per-task={r.cpus}",
        f"#SBATCH --mem={r.mem}",
        f"#SBATCH --time={r.time}",
        f"#SBATCH --output={out_dir}/slurm-%j.out",
        "",
        "set -euo pipefail",
        f"mkdir -p {out_dir}",
    ]
    for mod in manifest.env.module_load:
        lines.append(f"module load {mod}")
    if manifest.env.conda_env:
        lines.append('eval "$(conda shell.bash hook)"')
        lines.append(f"conda activate {manifest.env.conda_env}")
    for key, value in manifest.env.env.items():
        lines.append(f"export {key}={value}")
    if manifest.env.workdir:
        lines.append(f"cd {manifest.env.workdir}")
    lines += ["", command.strip(), ""]
    return "\n".join(lines)
