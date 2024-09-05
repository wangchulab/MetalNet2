import os.path as osp
import sys

utils_path = osp.join(osp.dirname(__file__),
                      "../../../../../dataset/collect/scripts/utils/")
sys.path.append(utils_path)

import pandas as pd
from absl import app, flags, logging
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from io_utils import parse_mmcif_file
from mmcif_utils import get_ndb_seq_can_num


def define_arguments():
    flags.DEFINE_string("input_pdb", None, required=True,
                        help="pdb id to (pdb file, resolution, deposition date), csv")
    flags.DEFINE_string("input_records", None, required=True,
                        help="records pdb, chain, csv")
    flags.DEFINE_string("output_records", None, required=True,
                        help="interaction file with residues, csv")
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn_type.id.html
    flags.DEFINE_string("conn_type", "disulf", help="interaction type")


def get_conn_residue_pairs(mmcif_dict: MMCIF2Dict):
    df = pd.DataFrame()

    df['conn_type'] = mmcif_dict['_struct_conn.conn_type_id']
    df['resi1_chain'] = mmcif_dict['_struct_conn.ptnr1_auth_asym_id']
    df['resi2_chain'] = mmcif_dict['_struct_conn.ptnr2_auth_asym_id']
    df['resi1'] = mmcif_dict['_struct_conn.ptnr1_label_comp_id']
    df['resi2'] = mmcif_dict['_struct_conn.ptnr2_label_comp_id']

    df['resi1_pdb_seq_num'] = mmcif_dict['_struct_conn.ptnr1_auth_seq_id']
    df['resi2_pdb_seq_num'] = mmcif_dict['_struct_conn.ptnr2_auth_seq_id']
    df['resi1_pdb_seq_num'] = df['resi1_pdb_seq_num'].astype(int)
    df['resi2_pdb_seq_num'] = df['resi2_pdb_seq_num'].astype(int)

    df['resi1_pdb_ins_code'] = mmcif_dict['_struct_conn.pdbx_ptnr1_PDB_ins_code']
    df['resi2_pdb_ins_code'] = mmcif_dict['_struct_conn.pdbx_ptnr2_PDB_ins_code']
    df['resi1_pdb_ins_code'] = df['resi1_pdb_ins_code'].map(
        lambda x: ' ' if x in {'.', '?'} else x)
    df['resi2_pdb_ins_code'] = df['resi2_pdb_ins_code'].map(
        lambda x: ' ' if x in {'.', '?'} else x)
    df['distance'] = mmcif_dict['_struct_conn.pdbx_dist_value']
    return df


def run(
    dict_pdb_file: dict,
    df: pd.DataFrame,
    conn_type: str,
) -> list:
    dfs = []
    # for each metal chain
    for pdb_chain, _ in df.groupby(['pdb', 'metal_chain']):
        pdb, chain = pdb_chain
        pdb_file = dict_pdb_file[pdb]
        try:
            _, mmcif_dict = parse_mmcif_file(pdb, pdb_file)
            df_pairs = get_conn_residue_pairs(mmcif_dict)
            ndb_seq_can_num_dict = get_ndb_seq_can_num(mmcif_dict)
        except:
            logging.error(
                f"Failed getting interacted residue pairs: {pdb_chain}")
            continue

        records = []
        for _, row in df_pairs.iterrows():

            # only consider conn_type involved in target_chain
            if not (
                row['conn_type'] == conn_type and
                row['resi1_chain'] == chain and
                row['resi2_chain'] == chain
            ):
                continue

            try:
                _, _, resi1_ndb_seq_can_num = ndb_seq_can_num_dict[(
                    row['resi1_chain'],
                    row['resi1_pdb_seq_num'],
                    row['resi1_pdb_ins_code'],
                    row['resi1']
                )]
                _, _, resi2_ndb_seq_can_num = ndb_seq_can_num_dict[(
                    row['resi2_chain'],
                    row['resi2_pdb_seq_num'],
                    row['resi2_pdb_ins_code'],
                    row['resi2']
                )]
                records.append({
                    'pdb': pdb,
                    'chain': chain,
                    'resi_ndb_can_seq_num_1': resi1_ndb_seq_can_num,
                    'resi_ndb_can_seq_num_2': resi2_ndb_seq_can_num,
                    'resi_pdb_seq_num_1': row['resi1_pdb_seq_num'],
                    'resi_pdb_seq_num_2': row['resi2_pdb_seq_num'],
                })
            except KeyError:
                continue
        dfs.append(pd.DataFrame(records))
    return dfs


def main(argv):
    FLAGS = flags.FLAGS
    input_pdb = FLAGS.input_pdb
    input_records = FLAGS.input_records
    output_records = FLAGS.output_records
    conn_type = FLAGS.conn_type

    df_pdb = pd.read_table(input_pdb)
    dict_pdb_file = dict(zip(df_pdb['pdb'], df_pdb['pdb_file']))
    df_input = pd.read_table(input_records)

    df_records = run(
        dict_pdb_file=dict_pdb_file,
        df=df_input,
        conn_type=conn_type
    )
    pd.concat(df_records).to_csv(output_records, sep="\t", index=None)


if __name__ == "__main__":
    define_arguments()
    app.run(main)
