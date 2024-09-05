import pandas as pd
from absl import app, flags
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SwissProt import FeatureTable

metal_elements = {
    'ZN', 'CA', 'MG', 'MN', 'FE',
    'CU', 'NI', 'CO',
    'SF4', 'FES', 'F3S'
}

def define_arguments():
    flags.DEFINE_string("input_swiss", None, required=True, help="swiss records file")
    flags.DEFINE_string("input_ligands", None, required=True, help="metal ligands, csv")
    flags.DEFINE_string("output", None, required=True, help="anotation file")
    
def annotate(
    swiss_file: str,
    chebi_to_metal: dict,
) -> pd.DataFrame:
    records = []
    for r in SeqIO.parse(swiss_file, "swiss"):
        r: SeqRecord
        features = r.features
        for f in features:
            f: FeatureTable
            if f.type == "BINDING":
                chebi = f.qualifiers['ligand_id'][6:] if 'ligand_id' in f.qualifiers.keys() else ' ' # ChEBI:CHEBI:...
                if chebi in chebi_to_metal.keys():
                    metal_resi = chebi_to_metal[chebi]
                    ligand_label = f.qualifiers['ligand_label'] if 'ligand_label' in f.qualifiers.keys() else 1
                    locations = list(f.location)
                    for posi in locations:
                        records.append({
                            "uniprot": r.id,
                            "resi_seq_num": posi + 1,
                            "resi": r.seq[posi],
                            "metal_resi": metal_resi,
                            "metal_chebi": chebi,
                            "metal_label": ligand_label,
                        })
    return pd.DataFrame(records)                           

def main(argv):
    FLAGS = flags.FLAGS
    input_swiss = FLAGS.input_swiss
    input_ligands = FLAGS.input_ligands
    output = FLAGS.output
    
    df_metal_ligands = pd.read_csv(input_ligands, sep="\t")
    df_metal_ligands = df_metal_ligands[df_metal_ligands['metal_resi'].map(lambda x: x in metal_elements)]
    dict_chebi_to_metal = dict(zip(df_metal_ligands['chebi'], df_metal_ligands['metal_resi']))
    df_anno = annotate(input_swiss, dict_chebi_to_metal)
    df_anno.to_csv(output, sep="\t", index=None)


if __name__ == "__main__":
    define_arguments()
    app.run(main)