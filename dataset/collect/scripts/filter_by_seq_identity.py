import os
from functools import cmp_to_key, partial
from typing import List, Tuple

import pandas as pd
from absl import app, flags, logging
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def define_arguments():
    flags.DEFINE_string("input_fasta", None, required=True, help="fasta file for cluster, pdb_chain as id")
    flags.DEFINE_string("input_pdb", None, required=True, help="pdb id to (pdb file, resolution, deposition date), csv")
    flags.DEFINE_string("input_records", None, required=True, help="metal binding sites file, csv")
    flags.DEFINE_string("output_records", None, required=True, help="non-redundant metal binding sites file, csv")
    flags.DEFINE_string("output_fasta", None, required=True, help="fasta file for nr metal binding chains")
    flags.DEFINE_string("output_pdb", None, required=True, help="pdb id to (pdb file, resolution, deposition date), csv")

def get_clusters(
    fasta_file: str,
    seq_id: float = 0.3
):
    fasta_file = os.path.realpath(fasta_file)
    os.mkdir("tmp/")
    os.chdir("tmp/")
    
    logging.info("Use mmseqs for clustering.")
    result = os.system(f"mmseqs easy-cluster {fasta_file} result tmp --min-seq-id {seq_id} -v 1")
    clusters = []
    if result == 0:
        df_cluster = pd.read_csv("./result_cluster.tsv", sep="\s+", header=None, names=['repr', 'member'])
        for _, df in df_cluster.groupby(by=['repr']):
            members = set(df['member'])
            clusters.append([tuple(m.split("_")) for m in members])
        logging.info("Done.")
    
    logging.info("Clean cluster files.")
    os.chdir("..")
    os.system("rm -rf tmp/")
    logging.info("Done.")
    return clusters

def select_chain_from_cluster(
    chain_x: tuple, # (pdb, chain)
    chain_y: tuple,
    dict_num_sites: dict, # (pdb, chain) -> num of sites
    dict_pdb_file: dict, # (pdb, chain) -> (pdb_file, resolution, deposition_date)
) -> int:
    num_sites_x, num_sites_y = dict_num_sites[chain_x], dict_num_sites[chain_y]
    if num_sites_x > num_sites_y: return 1
    elif num_sites_x < num_sites_y: return -1
    else:
        _, res_x, date_x = dict_pdb_file[chain_x[0]]
        _, res_y, date_y = dict_pdb_file[chain_y[0]]
        if res_x < res_y: return 1
        elif res_x > res_y: return -1
        else:
            if date_x > date_y: return 1
            elif date_x < date_y: return -1
            else: return 1 if chain_x > chain_y else -1

def filter_by_seq_identity(
    clusters: List[List[Tuple]], # (pdb, chain)
    dict_pdb_file: dict,
    df_sites: pd.DataFrame
):
    
    # calculate num of metal-binding sites for each pdb, chain
    dict_num_sites = dict() 
    for pdb_chain, df in df_sites.groupby(by=['pdb', 'metal_chain']):
        dict_num_sites[pdb_chain] = len(set(df['metal_pdb_seq_num']))
    
    # select pdb_chain with the maximum number of binding sites per cluster
    selected_nr_chains = set()
    for cluster in clusters:
        cluster = [i for i in cluster if i in dict_num_sites.keys()]
        if len(cluster) == 0:
            continue
        
        target_pdb_chain = max(cluster, key=cmp_to_key(partial(select_chain_from_cluster, dict_num_sites=dict_num_sites, dict_pdb_file=dict_pdb_file)))
        selected_nr_chains.add(target_pdb_chain)
        
    df_selected = df_sites[df_sites.apply(lambda row: (row['pdb'], row['metal_chain']) in selected_nr_chains, axis=1)]
    return df_selected

def main(argv):
    
    FLAGS = flags.FLAGS
    input_fasta = FLAGS.input_fasta
    input_pdb = FLAGS.input_pdb
    input_records = FLAGS.input_records
    output_records = FLAGS.output_records
    output_fasta = FLAGS.output_fasta
    output_pdb = FLAGS.output_pdb
    
    logging.info("Start selecting non redundant sites.")
    clusters = get_clusters(input_fasta)
    df_pdb = pd.read_csv(input_pdb, sep="\t")
    dict_pdb_files = dict(zip(df_pdb['pdb'], zip(df_pdb['pdb_file'], df_pdb['resolution'], df_pdb['deposition_date'])))
    df_output = filter_by_seq_identity(clusters, dict_pdb_files, pd.read_csv(input_records, sep="\t"))
    
    # output part
    ## records
    logging.info(f"Writing records to: {output_records}")
    df_output.to_csv(output_records, sep="\t", index=None)
    
    ## fasta
    seq_records = []
    metal_chains = set(zip(df_output['pdb'], df_output['metal_chain']))
    for r in SeqIO.parse(input_fasta, "fasta"):
        r: SeqRecord
        pdb, chain = r.id.split("_")
        if (pdb, chain) in metal_chains:
            seq_records.append(r)
    SeqIO.write(seq_records, output_fasta, "fasta")

    ## pdb files
    metal_pdbs = set(df_output['pdb'])
    df_pdb_output = df_pdb[df_pdb['pdb'].map(lambda x: x in metal_pdbs)]
    df_pdb_output.to_csv(output_pdb, sep="\t",)
    logging.info("Done.")
    

if __name__ == "__main__":
    define_arguments()
    app.run(main)