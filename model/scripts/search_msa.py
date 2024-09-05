import os
import os.path as osp
import traceback

import pandas as pd
from absl import app, flags, logging
from Bio import SeqIO
from utils.mmseqs import run_mmseqs2


def define_arguments():
    flags.DEFINE_string("input_fasta", None, required=True,
                        help="fasta with seq_id")
    flags.DEFINE_string("output_dir", None, required=True,
                        help="output msa dir")
    flags.DEFINE_string("output_msa", None, required=True,
                        help="output msa files, seq_id to msa_file, csv")
    flags.DEFINE_string("msa_type", "uniref",
                        help="msa type: merged, uniref, envdb")


def search_msa(
    seq_id: str,
    seq: str,
    output_file: str,
    msa_type: str,
):
    result = run_mmseqs2(seq, prefix=f"{seq_id}")
    msa_dir = f"{seq_id}_env"
    if msa_type == "uniref":
        os.system(f"cp {msa_dir}/uniref.a3m {output_file}")
        os.system(f'sed "s/\\x00//g" -i {output_file}')  # remove null char
    elif msa_type == "envdb":
        os.system(
            f"cp {msa_dir}/bfd.mgnify30.metaeuk30.smag30.a3m {output_file}")
        os.system(f'sed "s/\\x00//g" -i {output_file}')
    else:
        result = result[0]
        with open(output_file, 'w') as f:
            f.write(result)
    os.system(f"rm -rf {msa_dir}/")


def main(argv):
    FLAGS = flags.FLAGS
    input_fasta = FLAGS.input_fasta
    output_dir = FLAGS.output_dir
    output_msa = FLAGS.output_msa
    msa_type = FLAGS.msa_type

    logging.debug("Start searching msa...")
    logging.info(f"Using params msa type: {msa_type}")
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = osp.abspath(output_dir)

    msa_files = []
    for r in SeqIO.parse(input_fasta, "fasta"):
        seq_id, seq = r.id, str(r.seq)
        try:
            if msa_type == "uniref":
                msa_file = osp.join(output_dir, f"{seq_id}.uni.a3m")
            elif msa_type == "envdb":
                msa_file = osp.join(output_dir, f"{seq_id}.env.a3m")
            else:
                msa_file = osp.join(
                    output_dir, f"{seq_id}.mer.a3m")

            if not osp.exists(msa_file):
                search_msa(seq_id, seq, msa_file, msa_type)
            else:
                logging.info(f"Using existed msa file: {msa_file}")
            msa_files.append({
                "seq_id": seq_id,
                "msa_file": msa_file
            })
        except:
            logging.info(f"Failed searching msa for: {seq_id}")
            traceback.print_exc()
            continue
    pd.DataFrame(msa_files).to_csv(output_msa, sep="\t", index=None)
    logging.debug("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
