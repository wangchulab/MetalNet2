import os
import string
from typing import List, Tuple

import yaml
from Bio import SeqIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Structure import Structure
from sklearn.utils import Bunch

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def parse_a3m_file(a3m_file: str) -> List[str]:
    seqs = []
    a3m_file = os.path.expanduser(a3m_file)
    for record in SeqIO.parse(a3m_file, "fasta"):
        record: SeqIO.SeqRecord

        # delete lowercase characters and insertion characters from a string
        # see https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
        seq = str(record.seq).translate(translation)
        seqs.append(seq)
    return seqs


def parse_yaml_file(yaml_file: str) -> Bunch:

    yaml_file = os.path.expanduser(yaml_file)

    # NOTE: only the first layer of keys in yaml can be visited by value-mode
    d = yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader)
    return Bunch(**d)


def parse_mmcif_file(
    id: str,
    file: str,
) -> Tuple[Structure, MMCIF2Dict]:

    file = os.path.expanduser(file)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(id, file)
    mmcif_dict = parser._mmcif_dict
    return structure, mmcif_dict
