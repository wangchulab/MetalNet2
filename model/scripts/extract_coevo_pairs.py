import os
import os.path as osp
import traceback
from itertools import combinations
from typing import List

import esm
import numpy as np
import pandas as pd
import torch
from absl import app, flags, logging
from scipy.spatial.distance import cdist
from utils.io_utils import parse_a3m_file

ched = {'C', 'H', 'E', 'D'}
max_chain_length = 1023


def define_arguments():
    flags.DEFINE_string("input_msa", None, required=True,
                        help="seq_id to msa file (a3m or fasta format), csv")
    flags.DEFINE_string("output_pairs", None, required=True,
                        help="output coevolution files, csv")
    flags.DEFINE_string("msa_filter_type", "hamming",
                        help="msa filter method: hamming, hhfilter")
    flags.DEFINE_integer(
        "num_seq", 64, help="for max hamming, it's the maximum num of seqs retained; for hhfilter, it's the num of seq in -diff parameter")
    flags.DEFINE_float("coevo_threshold", 0.1,
                       help="value to define coevolution")
    flags.DEFINE_integer("cuda", None, help="the gpu device")


def load_msa_transformer(cuda: int):
    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    if cuda != None:
        device = torch.device(f"cuda:{cuda}")
        msa_transformer = msa_transformer.eval().to(device)
    else:
        msa_transformer = msa_transformer.eval()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    return msa_transformer, msa_batch_converter


@torch.no_grad()
def get_coevo_matrix(
    seqs: List[str],
    msa_transformer,
    msa_batch_converter,
    use_gpu: bool,
) -> torch.Tensor:
    # according to https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
    input_data = []
    for s in seqs:
        input_data.append(("_", s))
    _, _, msa_batch_tokens = msa_batch_converter(input_data)
    if use_gpu:
        msa_batch_tokens = msa_batch_tokens.to(
            next(msa_transformer.parameters()).device)
        msa_coevo_matrix = msa_transformer.predict_contacts(msa_batch_tokens)[
            0].cpu()
    else:
        msa_coevo_matrix = msa_transformer.predict_contacts(msa_batch_tokens)[
            0]
    return msa_coevo_matrix


def get_coevo_pairs(
    ref_seq: str,
    msa_coevo_matrix: torch.Tensor,
    coevo_threshold: float,
) -> pd.DataFrame:
    result = []
    for i, j in combinations(range(len(ref_seq)), 2):
        if abs(i - j) == 1:
            continue  # adjacent are not included
        coevo_value = msa_coevo_matrix[i][j]
        if coevo_value < coevo_threshold:
            continue
        resi_seq_posi_1 = i
        resi_seq_posi_2 = j
        resi_1 = ref_seq[i]
        resi_2 = ref_seq[j]
        if resi_1 not in ched or resi_2 not in ched:
            continue
        result.append({
            "resi_1": resi_1,
            # it's ndb_seq_can_posi (which is ndb_seq_can_num - 1), starts from 0
            "resi_seq_posi_1": resi_seq_posi_1,
            "resi_2": resi_2,
            "resi_seq_posi_2": resi_seq_posi_2,
            "coevo_value": float(coevo_value)
        })
    df = pd.DataFrame(result, columns=["resi_1", "resi_seq_posi_1", "resi_2", "resi_seq_posi_2", "coevo_value"])
    return df


def filter_msa_by_max_hamming(
    seqs: List[str],
    num_retained: int,
) -> List[str]:
    # according to https://doi.org/10.1101/2021.02.12.430858
    # according to https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
    if len(seqs) <= num_retained:
        return seqs

    array = np.array([list(seq) for seq in seqs],
                     dtype=np.bytes_).view(np.uint8)
    optfunc = np.argmax
    all_indices = np.arange(len(seqs))
    indices = [0]
    pairwise_distances = np.zeros((0, len(seqs)))
    for _ in range(num_retained - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(
            pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [seqs[idx] for idx in indices]


def filter_msa_by_hhfilter(
    msa_file: str,
    num_retained: int,
) -> List[str]:
    filtered_msa_file = f"{os.path.basename(msa_file)}.filtered"
    cmd = f"hhfilter -i {msa_file} -diff {num_retained} -cov 75 -id 90 -o {filtered_msa_file} -v 1"
    result = os.system(cmd)
    aln = []
    if result == 0:
        aln = parse_a3m_file(filtered_msa_file)
        os.remove(filtered_msa_file)
    return aln


def filter_msa(
    msa_file: str,
    num_seq: int,
    method: str,
) -> List[str]:
    # filter msa
    # NOTE: remove \x00 that may occur in msa file from colabfold server
    if method == "hhfilter":
        aln = filter_msa_by_hhfilter(msa_file, num_seq)
    elif method == "hamming":
        aln = parse_a3m_file(msa_file)
        aln = filter_msa_by_max_hamming(aln, num_seq)
    else:
        raise ValueError(f"Unexpected msa filter method: {method}")
    return aln


def main(argv):
    FLAGS = flags.FLAGS
    input_msa = FLAGS.input_msa
    output_pairs = FLAGS.output_pairs
    msa_filter_type = FLAGS.msa_filter_type
    coevo_threshold = FLAGS.coevo_threshold
    num_seq = FLAGS.num_seq
    cuda = FLAGS.cuda
    logging.info(f"Using params msa filter type: {msa_filter_type}")
    logging.info(f"Using params num of seq retained: {num_seq}")
    logging.info(f"Using params coevo thresholds: {coevo_threshold}")

    df_msa = pd.read_table(input_msa)
    # set up model (we don't want to reload it)
    msa_transformer, msa_batch_converter = load_msa_transformer(cuda)
    use_gpu = True if cuda != None else False

    # predcit coevo maps
    dfs = []
    for _, row in df_msa.iterrows():

        seq_id = row['seq_id']
        a3m_file = row['msa_file']
        if not osp.exists(a3m_file): # treat as relative path
            a3m_file = osp.join(osp.abspath(osp.dirname(input_msa)), a3m_file)

        logging.info(f"Start generating coevo pairs for {seq_id}")
        try:
            aln = filter_msa(a3m_file, num_seq, msa_filter_type)
        except:
            traceback.print_exc()
            logging.error(f"Failed parsing msa file {a3m_file}")
            continue
        target_seq = aln[0]
        if len(target_seq) > max_chain_length:
            logging.error(
                f"Sequence too long (>1023) for {seq_id} with length {len(target_seq)}.")
            continue
        coevo_mtx = None
        attempt_times = 3
        while True:
            try:
                if attempt_times <= 0:
                    logging.error(
                        f"Failed generating coevo matrix for {seq_id}")
                    break
                attempt_times -= 1
                coevo_mtx = get_coevo_matrix(
                    aln, msa_transformer, msa_batch_converter, use_gpu)
                break
            except:  # when gpu is out of memory
                logging.error(f"Renerating coevo matrix for {seq_id}")
                num_msa_reduced = int(len(aln) / 4) * 2
                aln = aln[:num_msa_reduced]
        if coevo_mtx != None:
            df = get_coevo_pairs(target_seq, coevo_mtx, coevo_threshold)
            df['seq_id'] = seq_id
            dfs.append(df)
    if len(dfs) > 0:
        df_output = pd.concat(dfs)
        # after concat, the type of seq_posi may change to float64 (in train pairs)
        df_output['resi_seq_posi_1'] = df_output['resi_seq_posi_1'].astype(int)
        df_output['resi_seq_posi_2'] = df_output['resi_seq_posi_2'].astype(int)
        df_output.to_csv(output_pairs, sep="\t", index=None)
    else:
        logging.error("No coevo pairs generated.")
    logging.info(f"Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
