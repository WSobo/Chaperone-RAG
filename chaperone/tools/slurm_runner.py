import subprocess
from langchain_core.tools import tool
from chaperone.utils.logger import logger

@tool
def submit_job(job_script_path: str) -> str:
    """
    Submits an sbatch job script to the SLURM HPC cluster schedule.
    Use this when you need to execute heavy computational tasks (like AlphaFold or RFdiffusion).
    """
    try:
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
    os.makedirs("scripts/", exist_ok=True)
    full_path = os.path.join("scripts", script_name)
    with open(full_path, "w") as f:
        f.write(script_content)
    # Ensure it's executable
    os.chmod(full_path, 0o755)
    return f"Successfully wrote executable script to {full_path}"

