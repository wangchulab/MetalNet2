import traceback

import pandas as pd
import requests
from absl import app, flags, logging
from Bio import SeqIO
from Bio.SwissProt import FeatureTable
from bs4 import BeautifulSoup

metal_elements = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 
    'K', 'Ca', 'Sc', 'Ti', 'V', 
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
    'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
    'In', 'Sn', 'Cs', 'Ba', 'La', 
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 
    'Er', 'Tm', 'Yb', 'Lu', 'Hf', 
    'Ta', 'W', 'Re', 'Os', 'Ir', 
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 
    'Bi', 'Fr', 'Ra', 'Ac', 'Th', 
    'Pa', 'U', 'Np', 'Pu', 'Am', 
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 
    'Md', 'No', 'Lr'
}

def define_arguments():
    flags.DEFINE_string("input", None, required=True, help="swiss records file")
    flags.DEFINE_string("output", None, required=True, help="ligands info file")

def get_ligands_from_swiss(swiss_file: str) -> pd.DataFrame:
    records = []
    for r in SeqIO.parse(swiss_file, 'swiss'):
        features = r.features
        for f in features:
            f: FeatureTable
            if f.type == "BINDING":
                ligand = f.qualifiers['ligand']
                chebi = f.qualifiers['ligand_id'][6:] if 'ligand_id' in f.qualifiers.keys() else ' ' # ChEBI:CHEBI:...
                records.append({
                    'chebi': chebi,
                    'ligand': ligand
                })
    return pd.DataFrame(records)
                
def get_inchi_from_chebi(chebi: str) -> str:
    chebi_url = f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi}"
    inchi = ' '
    try:
        inchi_response = requests.get(chebi_url, timeout=5.)
        lines = inchi_response.text.split("\n")
        for l in lines:
            line = l.strip()
            if line.startswith("<td>InChI"):
                soup = BeautifulSoup(line, 'lxml')  
                inchi = soup.get_text()
    except:
        traceback.print_exc()
        logging.error(f"Failed fetching inchi from chebi: {chebi}")
    return inchi

def has_metal_element(inchi: str):
    has_metal = False
    try: chem_formula = inchi[6:].split("/")[1] # InChI=...
    except: return False
    for i in metal_elements:
        if i in chem_formula:
            has_metal = True
            break
    return has_metal

def main(argv):
    FLAGS = flags.FLAGS
    input = FLAGS.input
    output = FLAGS.output
    
    logging.info(f"Start getting ligands from: {input}")
    df_ligands = get_ligands_from_swiss(input)
    df_ligands.drop_duplicates(['chebi'], inplace=True)
    df_ligands['inchi'] = df_ligands['chebi'].map(lambda x: get_inchi_from_chebi(x))
    df_ligands['has_metal'] = df_ligands['inchi'].map(lambda x: has_metal_element(x))
    
    df_ligands = df_ligands[df_ligands.apply(lambda row: row['chebi'] != ' ' and (row['has_metal'] == True or row['inchi'] == ' '), axis=1)]
    df_ligands.to_csv(output, sep="\t", index=None)
    logging.info("Done.")

if __name__ == "__main__":
    define_arguments()
    app.run(main)