import pandas as pd
from absl import app, flags, logging
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

ched = {'C', 'H', 'E', 'D'}
metal_name_translation = {"MN3": "MN", "CUA": "CU", "CU1": "CU", "UNL": "ZN", "FE2": "FE"} # after manully check, UNL belongs to ZN
max_chain_length = 1023


def define_arguments():
    flags.DEFINE_string("input_records", None, required=True, help="metal binding chains file, csv")
    flags.DEFINE_string("input_fasta", None, required=True, help="fasta file for cluster, pdb_chain as id")
    flags.DEFINE_string("output_train_records", None, required=True, help="metal binding train chains file with CHED only, csv")
    flags.DEFINE_string("output_test_records", None, required=True, help="metal binding test chains file with CHED only, csv")
    flags.DEFINE_string("output_fasta", None, required=True, help="fasta file, the length of protein chain is < 1024")


def domain_record_to_seq_id(record: tuple) -> str: 
    # (pdb, chain, domain) -> id
    pdb, chain, domain = record
    if domain == " ":
        return f"{pdb}_{chain}"
    else:
        lower, upper = eval(domain)
        return f"{pdb}_{chain}_{lower}_{upper}"
    
def main(argv):
    FLAGS = flags.FLAGS
    input_records = FLAGS.input_records
    input_fasta = FLAGS.input_fasta
    output_train_records = FLAGS.output_train_records
    output_test_records = FLAGS.output_test_records
    output_fasta = FLAGS.output_fasta
    
    logging.info("Start cleaning dataset for metalnet.")
    valid_domains = set()
    dict_domain_to_seq = dict()
    for r in SeqIO.parse(input_fasta, "fasta"):
        r: SeqRecord
        if len(r.seq) <= max_chain_length:
            valid_domains.add(r.id)
            dict_domain_to_seq[r.id] = r
    df_input = pd.read_csv(input_records, sep="\t")
    
    ## assign seq id, position in domain (start from 0), metal name
    df_output = df_input.copy()
    df_output['seq_id'] = df_output.apply(lambda row: domain_record_to_seq_id((row['pdb'], row['metal_chain'], row['domain'])), axis=1)
    df_output['resi_domain_posi'] = df_output.apply(lambda row: 
        row['resi_ndb_seq_can_num'] - 1 if row['domain'] == " "
        else row['resi_ndb_seq_can_num'] - eval(row['domain'])[0], 
    axis=1)
    df_output['metal_resi'] = df_output['metal_resi'].map(lambda x: metal_name_translation[x] if x in metal_name_translation.keys() else x)
    
    ## remove long domains, duplicate resi annotations and non-ched
    df_output = df_output[df_output['seq_id'].map(lambda x: x in valid_domains)]
    df_output = df_output.drop_duplicates(subset=['pdb', 'metal_chain', 'resi_ndb_seq_can_num'])
    df_output = df_output[df_output['resi'].map(lambda x: x in ched)]
    
    ## to output
    domains = set(zip(df_output['pdb'], df_output['metal_chain'], df_output['domain']))
    records = []
    for d in domains:
        id = domain_record_to_seq_id(d)
        records.append(dict_domain_to_seq[id])
        
    df_output_train = df_output[df_output['data_type'] == 'train_data']
    df_output_test = df_output[df_output['data_type'] == 'test_data']
    df_output_train = df_output_train.drop(columns=['data_type'])
    df_output_test = df_output_test.drop(columns=['data_type'])
    df_output_train.to_csv(output_train_records, sep="\t", index=None)
    df_output_test.to_csv(output_test_records, sep="\t", index=None)
    SeqIO.write(records, output_fasta, "fasta")
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)