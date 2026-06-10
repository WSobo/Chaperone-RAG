"""SLURM tools: author and submit HPC batch jobs (AlphaFold, RFdiffusion, ...)."""

from __future__ import annotations

import os
import subprocess

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chaperone.settings import get_settings
from chaperone.utils.logger import logger


class CreateScriptArgs(BaseModel):
    script_name: str = Field(description="Filename for the script, e.g. 'fold.sh'.")
    script_content: str = Field(description="Full bash/#SBATCH script body.")


class SubmitArgs(BaseModel):
    job_script_path: str = Field(description="Path to an existing sbatch script.")


@tool(args_schema=CreateScriptArgs)
def create_slurm_script(script_name: str, script_content: str) -> str:
    """Write an executable .sh sbatch script to the configured scripts dir.

    Use to formulate a SLURM job before submitting it with submit_job.
    """
    scripts_dir = str(get_settings().paths.scripts_dir)
    os.makedirs(scripts_dir, exist_ok=True)
    path = os.path.join(scripts_dir, script_name)
    with open(path, "w") as f:
        f.write(script_content)
    os.chmod(path, 0o755)
    logger.info(f"Wrote SLURM script -> {path}")
    return path


@tool(args_schema=SubmitArgs)
def submit_job(job_script_path: str) -> str:
    """Submit an sbatch job script to the SLURM scheduler; returns sbatch output.

    Use for heavy GPU compute (folding, diffusion) that shouldn't run inline.
    """
    try:
        result = subprocess.run(
            ["sbatch", job_script_path], capture_output=True, text=True, check=True
        )
        logger.info(f"Submitted: {result.stdout.strip()}")
        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: 'sbatch' not found (not on a SLURM login node)."
    except subprocess.CalledProcessError as e:
        logger.error(f"sbatch failed: {e.stderr}")
        return f"Error submitting job: {e.stderr}"
