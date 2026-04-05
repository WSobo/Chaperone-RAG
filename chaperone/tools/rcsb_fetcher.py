import os
import requests
from langchain_core.tools import tool
from chaperone.utils.logger import logger

@tool
def fetch_pdb_metadata(pdb_id: str) -> dict:
    """
    Fetches basic PDB metadata from the RCSB GraphQL API.
    Use this when you need basic information about a protein structure like title, method, or resolution.
    """
    pdb_id = pdb_id.upper()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        title = data.get("struct", {}).get("title", "No Title")
        method = data.get("exptl", [{}])[0].get("method", "Unknown Method")
        resolution = data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0]
        
        logger.info(f"Successfully fetched metadata for {pdb_id}")
        return {
            "pdb_id": pdb_id,
            "title": title,
            "method": method,
            "resolution": resolution,
            "url": f"https://www.rcsb.org/structure/{pdb_id}"
        }
    except Exception as e:
        logger.error(f"Failed to fetch {pdb_id} from RCSB: {e}")
        return {"error": str(e)}

@tool
def download_pdb_file(pdb_id: str, output_dir: str = "data/pdb_files/") -> str:
    """
    Downloads the full PDB file from RCSB to the local disk.
    Use this when you need the actual structural coordinates of a protein to analyze or run simulations on.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Downloaded PDB -> {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"Failed to download {pdb_id}: {e}")
        return ""
