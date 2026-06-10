"""RCSB PDB tools: structure metadata and coordinate download."""

from __future__ import annotations

import os

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chaperone.settings import get_settings
from chaperone.utils.logger import logger


class PdbIdArgs(BaseModel):
    pdb_id: str = Field(description="4-character RCSB PDB identifier, e.g. '1UBQ'.")


@tool(args_schema=PdbIdArgs)
def fetch_pdb_metadata(pdb_id: str) -> dict:
    """Fetch title, experimental method, and resolution for a PDB entry from RCSB.

    Use for quick facts about a known protein structure.
    """
    pdb_id = pdb_id.upper()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "pdb_id": pdb_id,
            "title": data.get("struct", {}).get("title", "No Title"),
            "method": (data.get("exptl") or [{}])[0].get("method", "Unknown"),
            "resolution": (data.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0],
            "url": f"https://www.rcsb.org/structure/{pdb_id}",
        }
    except requests.RequestException as e:
        logger.error(f"RCSB metadata fetch failed for {pdb_id}: {e}")
        return {"error": str(e)}


@tool(args_schema=PdbIdArgs)
def download_pdb_file(pdb_id: str) -> str:
    """Download the full .pdb coordinate file from RCSB to the configured pdb dir.

    Returns the local path, or an empty string on failure. Use when downstream
    analysis or simulation needs the actual atomic coordinates.
    """
    out_dir = str(get_settings().paths.pdb_dir)
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.lower()
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    try:
        resp = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", stream=True, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded PDB -> {out_path}")
        return out_path
    except requests.RequestException as e:
        logger.error(f"PDB download failed for {pdb_id}: {e}")
        return ""
