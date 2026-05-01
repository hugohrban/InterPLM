import os
import tempfile
from typing import Dict, List

import numpy as np
import py3Dmol
import requests
from Bio.PDB import PDBIO, PDBList, PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

from interplm.constants import PDB_DIR
from interplm.dashboard.colors import default_cyan_to_magenta_colormap

aa_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def parse_pdb_line(line):
    match = line.startswith("ATOM") and (line[13:15] == "CA")
    if match:
        residue_num = line[22:26].strip()
        residue_3 = line[17:20].strip()
        x, y, z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
        return {
            "residue_num": int(residue_num) - 1,
            "residue_letter": aa_3to1.get(residue_3, "X"),
            "coords": (float(x), float(y), float(z)),
        }
    return None


def get_single_chain_pdb_structure(pdb_id: str, chain_id: str):
    pdb_file = PDBList().retrieve_pdb_file(
        pdb_id, file_format="pdb", pdir="pdbs", overwrite=True
    )
    structure = PDBParser().get_structure("tmp", pdb_file)
    structure = structure[0][chain_id]
    return structure

def get_single_chain_afdb_structure(uniprot_id: str):
    os.makedirs(PDB_DIR, exist_ok=True)
    pdb_file_path = os.path.join(PDB_DIR, f"AF-{uniprot_id}-F1-model_v6.pdb")

    if not os.path.exists(pdb_file_path):
        api_response = requests.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=10
        )
        if api_response.status_code == 404:
            return None
        api_response.raise_for_status()

        pdb_url = api_response.json()[0]["pdbUrl"]
        pdb_response = requests.get(pdb_url, timeout=30)
        pdb_response.raise_for_status()
        with open(pdb_file_path, "w") as f:
            f.write(pdb_response.text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", pdb_file_path)
    return structure[0]


def get_structure_from_cif_file(cif_file_path: str, chain_id: str | None = None):
    """
    Parse a .cif file and return the structure.
    
    Args:
        cif_file_path: Path to the .cif file
        chain_id: Specific chain to extract. If None, returns the first chain.
    
    Returns:
        Bio.PDB structure object
    """
    if not os.path.exists(cif_file_path):
        raise FileNotFoundError(f"CIF file not found: {cif_file_path}")
    
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("cif_structure", cif_file_path)
    
    # Get the first model (index 0)
    model = structure[0]
    
    if chain_id is not None:
        # Return specific chain
        try:
            return model[chain_id]
        except KeyError:
            available_chains = [chain.id for chain in model]
            raise ValueError(f"Chain '{chain_id}' not found in CIF file. Available chains: {available_chains}")
    else:
        # Return the first chain
        chains = list(model)
        if chains:
            return chains[0]
        else:
            raise ValueError("No chains found in the CIF file")




def structure_to_seq(structure) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_pdb_file:
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_pdb_file.name)
        temp_pdb_file.seek(0)
        pdb_data = temp_pdb_file.read()
    return pdb_data


def default_colormap_fn(value):
    """Legacy default colormap function - kept for backwards compatibility."""
    if value > 0:
        return "magenta"
    else:
        return "cyan"



def view_single_protein(
    pdb_id: str | None = None,
    chain_id: str | None = None,
    uniprot_id: str | None = None,
    cif_file_path: str | None = None,
    values_to_color: List[float] | None = None,
    colormap_fn: callable = default_cyan_to_magenta_colormap,
    default_color: str = "lightgray",
    residues_to_highlight: List[int] | None = None,
    highlight_color: str = "magenta",
    pymol_params: Dict = {"width": 400, "height": 400},
) -> str:
    """
    Visualize a single protein structure with improved color assignment.
    
    Args:
        pdb_id: PDB ID to download from PDB database
        chain_id: Chain ID to extract (required if using pdb_id or cif_file_path)
        uniprot_id: UniProt ID to download from AlphaFold database
        cif_file_path: Path to local .cif file to load
        values_to_color: List of values to color residues by
        colormap_fn: Function to map values to colors
        default_color: Default color for residues
        residues_to_highlight: List of residue indices to highlight
        highlight_color: Color for highlighted residues
        pymol_params: Parameters for py3Dmol viewer
    
    Returns:
        HTML string for protein visualization
    """
    if cif_file_path is not None:
        pdb_struct = get_structure_from_cif_file(cif_file_path, chain_id)
        # If chain_id wasn't specified, get the chain id from the structure
        if chain_id is None:
            chain_id = pdb_struct.id
    elif uniprot_id is not None:
        pdb_struct = get_single_chain_afdb_structure(uniprot_id)
        if pdb_struct is None:
            raise ValueError(f"No AlphaFold structure found for UniProt ID: {uniprot_id}")
        chain_id = "A"
    elif pdb_id is not None and chain_id is not None:
        pdb_struct = get_single_chain_pdb_structure(pdb_id, chain_id)
    else:
        raise ValueError("Either pdb_id and chain_id, uniprot_id, or cif_file_path must be provided.")

    pdb_data = structure_to_seq(pdb_struct)
    residues = pdb_struct.get_residues()

    view = py3Dmol.view(**pymol_params)
    # view.setBackgroundColor(
    #    #"#0e1117"
    # )  # This is the streamlit dark theme background color
    view.addModel(pdb_data, "pdb")

    view.setStyle({"cartoon": {"color": default_color}})

    if values_to_color is None:
        values_to_color = [0] * len(residues)

    for res_id_in_seq, (residue, value) in enumerate(zip(residues, values_to_color)):
        color = colormap_fn(value)

        opacity = 0.95

        res_id_in_pdb = residue.id[1]
        residue_type = residue.get_resname()

        view.setStyle(
            {"chain": chain_id, "resi": res_id_in_pdb},
            {"cartoon": {"color": color, "opacity": opacity}},
        )

        if residues_to_highlight and res_id_in_seq in residues_to_highlight:
            view.addStyle(
                {"chain": chain_id, "resi": res_id_in_pdb},
                {"stick": {"color": highlight_color}},
            )
            view.addLabel(
                f"{residue_type}",  # {res_id_in_pdb}",
                {
                    "fontOpacity": 1,
                    "backgroundOpacity": 0.0,
                    "fontSize": 20,
                    "fontColor": highlight_color,
                },
                {"chain": chain_id, "resi": res_id_in_pdb},
            )

    view.zoomTo()
    return view._make_html()
