import os
from typing import Tuple

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Structure import Structure


def parse_mmcif_file(
    id: str,
    file: str,
) -> Tuple[Structure, MMCIF2Dict]:
    
    file = os.path.expanduser(file)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(id, file)
    mmcif_dict = parser._mmcif_dict
    return structure, mmcif_dict