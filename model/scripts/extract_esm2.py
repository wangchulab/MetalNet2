import os
import os.path as osp

import pandas as pd
from absl import app, flags, logging
from Bio import SeqIO


def define_arguments():
    flags.DEFINE_string("input_fasta", None, required=True,
                        help="fasta with seq_id")
    flags.DEFINE_string("output_dir", None, required=True,
                        help="output esm2 encodings dir")
    flags.DEFINE_string("output_esm2", None, required=True,
                        help="output esm2 files, seq_id to file, tsv")
    flags.DEFINE_string(
        "esm2_model", "~/database/model/esm2_t33_650M_UR50D.pt", help="esm2 model path (650M)")
    flags.DEFINE_integer("cuda", None, help="the gpu device")
    flags.DEFINE_integer("toks_per_batch", default=4096, help="maximum batch size")


def run(
    esm2_model: str,
    fasta_file: str,
    output_dir: str,
    cuda: int,
    toks_per_batch: int,
) -> int:

    esm2_script_file = osp.join(osp.dirname(
        osp.abspath(__file__)), "./utils/esm2.py")

    if cuda is not None:
        cmd = f'''
CUDA_VISIBLE_DEVICES={cuda} \
    python {esm2_script_file} \
    {esm2_model} \
    {fasta_file} \
    {output_dir} \
    --toks_per_batch {toks_per_batch} \
    --truncation_seq_length 2700 \
    --include per_tok
'''
    else:
        cmd = f'''
python {esm2_script_file} \
    {esm2_model} \
    {fasta_file} \
    {output_dir} \
    --toks_per_batch {toks_per_batch} \
    --truncation_seq_length 2700 \
    --include per_tok \
    --nogpu        
'''

    return os.system(cmd)


def main(argv):
    FLAGS = flags.FLAGS
    input_fasta = FLAGS.input_fasta
    output_dir = FLAGS.output_dir
    output_esm2 = FLAGS.output_esm2
    esm2_model = FLAGS.esm2_model
    cuda = FLAGS.cuda
    toks_per_batch = FLAGS.toks_per_batch

    logging.info("Start extracting esm2 encoding...")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    result = run(
        esm2_model=esm2_model,
        fasta_file=input_fasta,
        output_dir=output_dir,
        cuda=cuda,
        toks_per_batch=toks_per_batch
    )

    if result == 0:
        esm2_files = []
        seq_ids = []
        for r in SeqIO.parse(osp.expanduser(input_fasta), "fasta"):
            seq_ids.append(r.id)

        for id in seq_ids:
            encoding_file = osp.abspath(osp.join(output_dir, f"{id}.pt"))
            if os.path.exists(encoding_file):
                esm2_files.append({
                    "seq_id": id,
                    "file": encoding_file
                })
        logging.info(
            f"Successfully extracting esm2 encodings ({len(esm2_files)} / {len(seq_ids)})")
        pd.DataFrame(esm2_files).to_csv(output_esm2, sep="\t", index=None)
    else:
        logging.error("Failed extracting esm2.")
    logging.debug("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
