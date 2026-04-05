import os
import datetime
import hashlib
from pathlib import Path
from typing import List, Dict
from langchain_core.tools import tool
from chaperone.utils.logger import logger

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@tool
def generate_reproducibility_bundle(
    experiment_name: str, 
    python_scripts: Dict[str, str], 
    bash_commands: List[str], 
    conda_deps: List[str], 
    report_md: str
) -> str:
    """
    Creates a strict, reproducible experiment bundle containing scripts, environments, and logs.
    ALWAYS use this tool when you write code to run a simulation, structure prediction, or calculate metrics.
    
    Args:
        experiment_name: Name of the folder to create.
        python_scripts: Dictionary mapping filenames (e.g. 'run_af3.py') to their string code content.
        bash_commands: List of bash commands that were executed (or should be executed) to run the experiment.
        conda_deps: List of conda dependencies required (e.g. ['biopython', 'numpy']).
        report_md: A final markdown report summarizing the findings, methods, and results.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{experiment_name}_{timestamp}"
    output_dir = Path("data/experiments") / bundle_name
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write the markdown report
        (output_dir / "report.md").write_text(report_md)
        
        # 2. Write Python Scripts
        for fname, content in python_scripts.items():
            script_path = output_dir / fname
            script_path.write_text(content)
            
        # 3. Write Bash Execution script
        sh_content = f"#!/bin/bash\n# Chaperone-RAG Experiment: {experiment_name}\n# Generated: {timestamp}\nset -euo pipefail\n\n"
        sh_content += "\n".join(bash_commands) + "\n"
        cmd_path = output_dir / "run.sh"
        cmd_path.write_text(sh_content)
        os.chmod(cmd_path, 0o755)
        
        # 4. Write Conda Environment
        env_content = f"name: chaperone_{experiment_name}\nchannels:\n  - conda-forge\n  - bioconda\ndependencies:\n  - python>=3.10\n"
        env_content += "".join(f"  - {d}\n" for d in conda_deps)
        (output_dir / "environment.yml").write_text(env_content)
        
        # 5. Generate Checksums
        checksums = []
        for fpath in sorted(output_dir.rglob("*")):
            if fpath.is_file() and fpath.name != "checksums.sha256":
                rel = fpath.relative_to(output_dir)
                checksums.append(f"{_sha256_file(fpath)}  {rel}")
        (output_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n")
        
        logger.info(f"Reproducibility bundle created at {output_dir}")
        return f"Successfully generated reproducibility bundle at: {output_dir.absolute()}"
        
    except Exception as e:
        logger.error(f"Failed to generate reproducibility bundle: {e}")
        return f"Error creating bundle: {e}"