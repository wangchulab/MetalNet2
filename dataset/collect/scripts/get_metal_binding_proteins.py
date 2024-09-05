import multiprocessing as mp
from typing import List, Tuple

import pandas as pd
from absl import app, flags, logging
from Bio import SeqIO
from Bio.PDB.Atom import Atom
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Model import Model
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from more_itertools import divide
from utils.constants import common_amino_acids, metal_elements
from utils.io_utils import parse_mmcif_file
from utils.mmcif_utils import get_ndb_seqs_can


def define_arguments() -> None:
    flags.DEFINE_string("input", None, required=True, help="pdb id to (pdb file), csv")
    flags.DEFINE_string("output_fasta", None, required=True, help="metal binding sites file, fasta")
    flags.DEFINE_string("output_records", None, required=True, help="pdb id to (pdb file, resolution, deposition date), csv")
    flags.DEFINE_float("resolution_threshold", None, help="resolution threshold for pdb")
    
def is_metal_binding_protein(
    model: Model,
    
    # these are very loose conditions to define a metal binding protein
    # so you may not need to rerun this script when changing the downstream filter conditions
    interacted_distance = 4, # S in cysteine, with a covalent radius 1 Å; Cs, 2.6 Å. From https://en.wikipedia.org/wiki/Atomic_radius
    interacted_num_resi = 1, # num of common amino acids interacted with metal
):
    is_metal_binding = False

    # first: find metal residues and all atoms in standard aa
    aa_atoms = []
    metal_residues = []
    for r in model.get_residues():
        r: Residue
        hetflag: str = r.get_id()[0]
        
        # hetero residues
        if hetflag.startswith("H_"):
            for a in r.get_atoms():
                a: Atom
                # is a metal residue
                if str.capitalize(a.element) in metal_elements:
                    metal_residues.append(r)
                    break
                    
        elif hetflag == " ":
            # all atoms in standard aa
            if str.capitalize(r.get_resname()) in common_amino_acids:
                aa_atoms.extend([a for a in r.get_atoms()])
    if len(metal_residues) == 0 or len(aa_atoms) == 0:
        return is_metal_binding
    
    # second: search metal binding residues
    ns = NeighborSearch(aa_atoms)
    for r in metal_residues:
        r: Residue
        binding_residues = set()
        has_metal_binding_residue = False
        for a in r.get_atoms():
            a: Atom
            
            # one metal residue may have multiple metal ions
            # so we find all metal binding residues to these ions
            if str.capitalize(a.element) in metal_elements:
                binding_residues = binding_residues | set(ns.search(a.get_vector().get_array(), interacted_distance, "R"))
                if len(binding_residues) >= interacted_num_resi:
                    has_metal_binding_residue = True
                    break
        
        if has_metal_binding_residue:
            is_metal_binding = True
            break
    
    return is_metal_binding

def get_metal_binding_protein_info(
    pdb_id: str,
    pdb_file: str,
    resolution_threshold: float = 3., # usually it's 3 Å, see https://doi.org/10.1155/2022/8965712
    interacted_distance = 4,
    interacted_num_resi = 1,
) -> Tuple[dict, List[SeqRecord]]: # pdb record and seqs (pdb_chain to seq)
    
    record = {}
    seqs = []
    
    structure: Structure = None
    mmcif_dict: MMCIF2Dict = None
    try: structure, mmcif_dict = parse_mmcif_file(pdb_id, pdb_file)
    except: return record, seqs
    
    # check resolution
    resolution = structure.header['resolution'] # resolutions of some cryo-em structures can not be correctly parsed in biopython 1.81, such as 5vrf
    depostion_date = structure.header['deposition_date']
    if resolution_threshold != None:
        if not (resolution and resolution > 0 and resolution <= resolution_threshold):
            return record, seqs
    
    # check if it's a metal-binding protein
    model: Model = next(structure.get_models()) # to simplyfy, choose the first model
    is_metal_binding = is_metal_binding_protein(model, interacted_distance, interacted_num_resi)
    if is_metal_binding:
        record = {
            "pdb": pdb_id,
            "pdb_file": pdb_file,
            "resolution": resolution,
            "deposition_date": depostion_date
        }
        ndb_seqs_can = get_ndb_seqs_can(mmcif_dict)
        seqs = [SeqRecord(Seq(seq), id=f"{pdb_id}_{chain}", description="") for chain, seq in ndb_seqs_can.items()]
    
    return record, seqs

def run(
    dict_pdb_file: dict,
    **kwargs,
):
    records = []
    seqs = []
    for pdb_id, pdb_file in dict_pdb_file.items():
        try:
            record, seqs_ = get_metal_binding_protein_info(pdb_id, pdb_file, **kwargs)
            if len(seqs_) > 0:
                records.append(record)
                seqs.extend(seqs_)
        except Exception as e:
            logging.error(f"{e}")
            logging.error(f"Failed in protein with pdb id: {pdb_id}")
    
    return records, seqs

def main(argv):
    
    FLAGS = flags.FLAGS
    input = FLAGS.input
    output_records = FLAGS.output_records
    output_fasta = FLAGS.output_fasta
    resolution_threshold = FLAGS.resolution_threshold
    
    df_pdb = pd.read_csv(input, sep="\t")

    # divide into groups and multi process the task
    logging.info("Start finding metal binding sites.")
    logging.info(f"Using params resolution threshold: {resolution_threshold}")
    num_task = mp.cpu_count()
    pdb_groups = [dict(list(pg)) for pg in divide(num_task, dict(zip(df_pdb['pdb'], df_pdb['pdb_file'])).items())]
    jobs = mp.Pool(num_task)
    result = [jobs.apply_async(run, args=(pg,), kwds={'resolution_threshold': resolution_threshold}) for pg in pdb_groups]
    result = [r.get() for r in result]

    records = []
    seqs = []
    for r in result:
        records.extend(r[0])
        seqs.extend(r[1])
    
    pd.DataFrame(records).to_csv(output_records, sep="\t", index=None)
    SeqIO.write(seqs, output_fasta, "fasta")
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)