import os
import subprocess
from langchain_core.tools import tool
from chaperone.utils.logger import logger

@tool
def execute_python_script(script_name: str, code: str) -> str:
    """
    Writes and executes a Python script in a controlled workspace directory.
    Use this to perform active bioinformatics calculations, run BioPython scripts, 
    or format data on the fly.
    """
    workspace = "data/sandbox"
    os.makedirs(workspace, exist_ok=True)
    
    script_path = os.path.join(workspace, script_name)
    with open(script_path, "w") as f:
        f.write(code)
        
    logger.info(f"Executing sandbox script: {script_path}")
    try:
        # Run the script with a timeout to avoid infinite loops
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=True
        )
        return f"Execution successful.\nSTDOUT:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        logger.error(f"Sandbox script failed: {e.stderr}")
        return f"Execution failed with error.\nSTDERR:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        logger.error("Sandbox script timed out!")
        return "Execution timed out after 120 seconds."
