"""Typed bio tools for the agent (Pydantic argument schemas, LangChain @tool)."""

from chaperone.tools.literature import search_literature, web_search
from chaperone.tools.rcsb import download_pdb_file, fetch_pdb_metadata
from chaperone.tools.sandbox import execute_python_script
from chaperone.tools.slurm import create_slurm_script, submit_job

# Registered toolset — the agent's single wiring point.
ALL_TOOLS = [
    fetch_pdb_metadata,
    download_pdb_file,
    search_literature,
    web_search,
    create_slurm_script,
    submit_job,
    execute_python_script,
]

__all__ = [
    "ALL_TOOLS",
    "create_slurm_script",
    "download_pdb_file",
    "execute_python_script",
    "fetch_pdb_metadata",
    "search_literature",
    "submit_job",
    "web_search",
]
