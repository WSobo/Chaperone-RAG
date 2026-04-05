import subprocess
import re
from langchain_core.tools import tool
from chaperone.utils.logger import logger

SAFE_SCRIPT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _is_safe_script_path(path: str) -> bool:
    normalized = path.strip()
    if not normalized.endswith(".sh"):
        return False
    if normalized.startswith("/"):
        return False
    if ".." in normalized:
        return False
    return normalized.startswith("scripts/")

@tool
def submit_job(job_script_path: str) -> str:
    """
    Submits an sbatch job script to the SLURM HPC cluster schedule.
    Use this when you need to execute heavy computational tasks (like AlphaFold or RFdiffusion).
    """
    try:
        if not _is_safe_script_path(job_script_path):
            return "Error submitting job: only relative scripts/*.sh paths are allowed."

        result = subprocess.run(
            ['sbatch', job_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Job submitted successfully: {result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit SLURM job: {e.stderr}")
        return f"Error submitting job: {e.stderr}"

@tool
def create_slurm_script(script_name: str, script_content: str) -> str:
    """
    Creates a .sh script file on disk directly from a string of bash/sbatch commands.
    Use this to formulate SLURM jobs before calling submit_job.
    """
    import os
    clean_name = script_name.strip()
    if not SAFE_SCRIPT_RE.match(clean_name):
        return "Failed to create script: script_name contains invalid characters."
    if not clean_name.endswith(".sh"):
        clean_name = f"{clean_name}.sh"

    os.makedirs("scripts/", exist_ok=True)
    full_path = os.path.join("scripts", clean_name)
    with open(full_path, "w") as f:
        f.write(script_content)
    # Ensure it's executable
    os.chmod(full_path, 0o755)
    return f"Successfully wrote executable script to {full_path}"

