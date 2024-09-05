import multiprocessing as mp
from typing import Set, Tuple

import pandas as pd
from absl import app, flags, logging
from Bio.PDB.Atom import Atom
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Model import Model
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Residue import Residue
from more_itertools import divide
from utils.constants import *
from utils.io_utils import parse_mmcif_file
from utils.mmcif_utils import get_ndb_seq_can_num, get_ndb_seqs_can


def define_arguments():
    flags.DEFINE_string("input", None, required=True, help="pdb id to (pdb file, resolution, deposition date), csv")
    flags.DEFINE_string("output", None, required=True, help="metal binding sites file, csv")
    flags.DEFINE_float("interacted_distance", 3., help="threshold distance for metal-resi interactioin")
    flags.DEFINE_integer("interacted_num_resi", 3, help="threshold num for coordinate residues")
    flags.DEFINE_bool("include_common_metal_only", False, help="for single metal ion, only consider Zn, Ca, Mg, Mn, Fe, Cu, Ni, Co")
    flags.DEFINE_bool("include_metal_compound", False, help="take SF4, FES and F3S into consideration")
    flags.DEFINE_bool("include_main_chain", False, help="allow main chain O and N atom as ligand")
    flags.DEFINE_bool("include_water", False, help="allow water as ligand")
    flags.DEFINE_bool("use_sloppy_mode", False, help="sloppy mode, if True, do not consider has_coordinate_residue_on_metal_chain, \
        unexpected_coordinate_residue and disordered_entities, default as False")

def select_metal_binding_sites(
    pdb_id: str,
    pdb_file: str,
    interacted_distance: float = 3.0,
    interacted_num_resi: int = 3,
    include_common_metal_only: bool = True,
    include_metal_compound: bool = True,
    include_main_chain: bool = True,
    include_water: bool = True,
    chain_length_threshold: int = 20,
    use_sloppy_mode: bool = False,
):
    
    records = []
    
    m: Model = None
    mmcif_dict: MMCIF2Dict = None
    try: 
        structure, mmcif_dict = parse_mmcif_file(pdb_id, pdb_file)
        m = next(structure.get_models()) # to simplyfy, choose the first model
    except: return records
    
    # first: find metal residues and collect atoms for neighbors search
    ## only two type of metal residues are allowed
    ## 1. single metal ion (only one kind of metal element, but probably multi metal atoms) 
    ## 2. specific metal compound (defined in common_metal_compound_names, if include_metal_compound)
    atoms = [] 
    metal_residues = []
    for r in m.get_residues():
        r: Residue
        hetflag: str = r.get_id()[0]
        
        if hetflag == " ":
            if str.capitalize(r.get_resname()) in common_amino_acids:
                ## these atoms are legal coordinate atoms
                if include_main_chain: atoms.extend([a for a in r.get_atoms() if a.element in hetero_atoms])
                else: atoms.extend([a for a in r.get_atoms() if a.element in hetero_atoms and a.get_name() not in main_chain_atom_names])
            else: atoms.extend([a for a in r.get_atoms()])
        
        ## hetero residues ("H_" may also include water as "H_HOH") or water (as "W")
        else:
            atoms.extend([a for a in r.get_atoms()])
            elements = set([str.capitalize(a.element) for a in r.get_atoms()])
            if len(elements & metal_elements) != 0:
                if len(elements) == 1:
                    element = list(elements)[0]
                    if element in (common_metal_elements if include_common_metal_only else metal_elements):
                        metal_residues.append(r) # first allowed metal reisude type
                else:
                    if hetflag[2:] in common_metal_compound_names:
                        if include_metal_compound:
                            metal_residues.append(r) # second allowed metal reisude type
               
    # second: search metal binding residues
    ns = NeighborSearch(atoms)
    metal_to_coordinate_atoms = dict()
    ndb_seqs_can = get_ndb_seqs_can(mmcif_dict)
    for mr in metal_residues:
        mr: Residue
        
        # check the length of protein chain whose id is the same as chain_id of metal residue (may raise error if there's no such chain)
        # falied cases: ('2jf9', 'R'), ('2zvl', 'Z'), ('6yo5', 'GGG'), ('7ako', 'C')
        ndb_seq_can = ndb_seqs_can[mr.get_full_id()[2]]
        if len(ndb_seq_can) < chain_length_threshold:
            continue

        # one metal residue may have multiple metal ions
        # so we find all metal coordinate residues to these ions
        coordinate_residues = set()
        valid_coordinate_atoms = set() # (atom, atom distance)
        for ma in mr.get_atoms():
            ma: Atom
            if str.capitalize(ma.element) in metal_elements:
                atoms = set(ns.search(ma.get_vector().get_array(), interacted_distance, "A"))
                for a in atoms:
                    a: Atom
                    r: Residue = a.get_parent()
                    hetflag: str = r.get_id()[0]
                    coordinate_residues.add(r)
                    if hetflag == " " and str.capitalize(r.get_resname()) in common_amino_acids: valid_coordinate_atoms.add((a, a - ma))
        
        # check if:
        # 1. do not have unexpected coordinate residue (such as solvent or nucleotide or other canonical amino acids)
        # 2. at least one coordinate residue lies on the metal chain (removing redundancy is based on the seq of metal chain), e.g., 1ff5
        # 3. all metal binding residues are not disordered, e.g., 1ivu, 6dtk
        has_unexpected_coordinate_residue = False
        has_coordinate_residue_on_metal_chain = False
        has_disordered_entities = False
        valid_coordinate_residues = set()
        for r in coordinate_residues:
            r: Residue
            hetflag: str = r.get_id()[0]
            
            # check first part
            if hetflag != " ":
                if hetflag == "W" or hetflag == "H_HOH":
                    if not include_water: has_unexpected_coordinate_residue = True
                else:
                    if r != mr: has_unexpected_coordinate_residue = True
            else:
                if str.capitalize(r.get_resname()) not in common_amino_acids: has_unexpected_coordinate_residue = True
                else:
                    # check second part
                    if r.get_full_id()[2] == mr.get_full_id()[2]: has_coordinate_residue_on_metal_chain = True
                    # check thrid part
                    if r.is_disordered(): has_disordered_entities = True
                    valid_coordinate_residues.add(r) # if use sloppy mode, disordered residues are counted

        if not use_sloppy_mode:
            if not has_coordinate_residue_on_metal_chain: continue
            if has_disordered_entities or has_unexpected_coordinate_residue: continue
        
        if len(valid_coordinate_residues) >= interacted_num_resi: 
            metal_to_coordinate_atoms[mr] = valid_coordinate_atoms
    
    # third: to records
    ndb_seq_can_num_dict = get_ndb_seq_can_num(mmcif_dict)
    for k, v in metal_to_coordinate_atoms.items():
        
        k: Residue
        v: Set[Tuple[Atom, float]]

        pdb, model, chain, metal_info = k.get_full_id()
        metal_name, metal_pdb_seq_num, _ = metal_info
        
        for atom, distance in v:
            atom: Atom
            r: Residue = atom.get_parent()
            
            resi_name = r.get_resname()
            resi_chain = r.get_full_id()[2]
            resi_pdb_seq_num = r.get_id()[1]
            resi_pdb_ins_code = r.get_id()[2]
            _, resi_ndb_seq_num, resi_ndb_seq_can_num = ndb_seq_can_num_dict[(resi_chain, resi_pdb_seq_num, resi_pdb_ins_code, resi_name)]
            
            record = {
                'pdb': pdb,
                'model': model,
                'metal_chain': chain, # chain id for metal
                'metal_pdb_seq_num': metal_pdb_seq_num,
                'metal_resi': metal_name[2:], # 'H_ZN' to 'ZN'
                'resi_chain': resi_chain,
                'resi_pdb_seq_num': resi_pdb_seq_num,
                'resi_ndb_seq_num': resi_ndb_seq_num,
                'resi_ndb_seq_can_num': resi_ndb_seq_can_num,
                'resi_pdb_ins_code': resi_pdb_ins_code,
                'resi': str.capitalize(resi_name),
                'atom': atom.get_name(),
                'distance': distance
            }
            records.append(record)
    return records

def run(
    dict_pdb_file: dict,
    **kwargs,
):
    records = []
    for pdb_id, pdb_file in dict_pdb_file.items():
        try: 
            records.extend(select_metal_binding_sites(pdb_id, pdb_file, **kwargs))
        except:
            logging.error(f"Failed selecting metal binding sites in protein with pdb id: {pdb_id}")
    
    return records

def main(argv):
    FLAGS = flags.FLAGS
    input = FLAGS.input
    output = FLAGS.output
    interacted_distance = FLAGS.interacted_distance
    interacted_num_resi = FLAGS.interacted_num_resi
    include_common_metal_only = FLAGS.include_common_metal_only
    include_metal_compound = FLAGS.include_metal_compound
    include_main_chain = FLAGS.include_main_chain
    include_water = FLAGS.include_water
    use_sloppy_mode = FLAGS.use_sloppy_mode
    
    logging.info("Start selecting metal binding sites.")
    logging.info(f"Using params interacted_distance: {interacted_distance}")
    logging.info(f"Using params interacted_num_resi: {interacted_num_resi}")
    logging.info(f"Using params include_common_metal_only: {include_common_metal_only}")
    logging.info(f"Using params include_metal_compound: {include_metal_compound}")
    logging.info(f"Using params include_main_chain: {include_main_chain}")
    logging.info(f"Using params include_water: {include_water}")
    logging.info(f"Using params use_sloppy_mode: {use_sloppy_mode}")
    
    df_pdb = pd.read_csv(input, sep="\t")
    num_task = mp.cpu_count()
    pdb_groups = [dict(list(pg)) for pg in divide(num_task, dict(zip(df_pdb['pdb'], df_pdb['pdb_file'])).items())]
    jobs = mp.Pool(num_task)
    result = [jobs.apply_async(run, args=(pg,), kwds={
        "interacted_distance": interacted_distance,
        "interacted_num_resi": interacted_num_resi,
        "include_common_metal_only": include_common_metal_only,
        "include_metal_compound": include_metal_compound,
        "include_main_chain": include_main_chain,
        "include_water": include_water,
        "use_sloppy_mode": use_sloppy_mode,
    }) for pg in pdb_groups]
    result = [r.get() for r in result]

    records = []
    for r in result:
        records.extend(r)
    logging.info(f"Writing metal-biding sites to: {output}")
    pd.DataFrame(records).to_csv(output, sep="\t", index=None)
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)