import multiprocessing as mp

import pandas as pd
from absl import app, flags, logging
from Bio.SeqUtils import seq1
from utils.data_utils import split_data_frame_by_id
from utils.io_utils import parse_mmcif_file
from utils.mmcif_utils import get_ndb_seqs_can


def define_arguments():
    flags.DEFINE_string("input_records", None, required=True, help="metal binding sites file, csv")
    flags.DEFINE_string("input_pdb", None, required=True, help="pdb id to pdb file path, csv")
    flags.DEFINE_string("output_records", None, required=True, help="metal binding chains file, csv")

def run(
    dict_pdb_file: dict,
    df: pd.DataFrame
):
    records = []
    for pdb_chain, df_chain in df.groupby(['pdb', 'metal_chain']): # for each metal chain
        pdb, chain = pdb_chain
        
        pdb_file = dict_pdb_file[pdb]
        _, mmcif_dict = parse_mmcif_file(pdb, pdb_file)
        ndb_seqs_can = get_ndb_seqs_can(mmcif_dict)
        homo_chains = set([k for k, v in ndb_seqs_can.items() if v == ndb_seqs_can[chain]]) # chains with identical sequence
        for _, row in df_chain.iterrows():
            if row['resi_chain'] in homo_chains:
                records.append({
                    'pdb': pdb,
                    'metal_chain': chain,
                    'metal_pdb_seq_num': row['metal_pdb_seq_num'],
                    'metal_resi': row['metal_resi'],
                    'resi_chain': row['resi_chain'], # for subsequent analysis
                    'resi_ndb_seq_can_num': row['resi_ndb_seq_can_num'],
                    'resi': seq1(row['resi']),
                    'resi_pdb_id': (" ", row['resi_pdb_seq_num'], row['resi_pdb_ins_code'])
                })
    return records
            
def main(argv):
    FLAGS = flags.FLAGS
    input_pdb = FLAGS.input_pdb
    input_records = FLAGS.input_records
    output_records = FLAGS.output_records

    logging.info("Start transferring annotations.")
    df_pdb = pd.read_csv(input_pdb, sep="\t")
    dict_pdb_file = dict(zip(df_pdb['pdb'], df_pdb['pdb_file']))
    df_input = pd.read_csv(input_records, sep="\t")
    
    num_task = mp.cpu_count()
    df_chains = split_data_frame_by_id(df_input, id=['pdb', 'metal_chain'], n=num_task)
    jobs = mp.Pool(num_task)
    result = [jobs.apply_async(run, args=(
        dict_pdb_file, pg
    )) for pg in df_chains]
    result = [r.get() for r in result]

    records = []
    for r in result:
        records.extend(r)
    df_output = pd.DataFrame(records).drop_duplicates(
        subset=['pdb', 'metal_chain', 'metal_pdb_seq_num', 'resi_chain', 'resi_ndb_seq_can_num']
    ) # one residue may have multiple atoms that interact with metal
    df_output.to_csv(output_records, sep="\t", index=None)
    logging.info("Done.")

if __name__ == "__main__":
    define_arguments()
    app.run(main)
