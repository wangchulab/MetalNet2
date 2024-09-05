import math
import multiprocessing as mp
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from absl import app, flags, logging
from Bio import SeqIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from utils.data_utils import split_data_frame_by_id
from utils.io_utils import parse_mmcif_file
from utils.mmcif_utils import get_ndb_seq_can_num, get_ndb_seqs_can
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def define_arguments():
    flags.DEFINE_string("input_records", None, required=True,
                        help="metal binding sites file, csv")
    flags.DEFINE_string("input_pdb", None, required=True,
                        help="pdb id to pdb file path, csv")
    flags.DEFINE_string("input_fasta", None, required=True,
                        help="fasta file, pdb_chain as id")
    flags.DEFINE_string("output_records", None, required=True,
                        help="metal binding chains file with assigned domain, csv")
    flags.DEFINE_string("output_fasta", None, required=True,
                        help="fasta file for metal binding chains with domains")


def get_domain_boundary_score(
    pdb_id: str,
    pdb_file: str,
    chain_id: str,
    interaction_threshold: float = 8.,
) -> dict:  # ndb_seq_can_num -> score

    structure, mmcif_dict = parse_mmcif_file(pdb_id, pdb_file)
    model = next(structure.get_models())

    ndb_seq_can_num_to_residue = dict()
    ndb_seq_can_length = None
    for c in model.get_chains():
        c: Chain
        if c.get_full_id()[2] == chain_id:
            residues = c.get_residues()
            ndb_seq_can_num_dict = get_ndb_seq_can_num(mmcif_dict)
            ndb_seq_can_length = len(get_ndb_seqs_can(mmcif_dict)[chain_id])
            for r in residues:
                r: Residue
                resi_name = r.get_resname()
                resi_chain = r.get_full_id()[2]
                resi_pdb_seq_num = r.get_id()[1]
                resi_pdb_ins_code = r.get_id()[2]
                try:
                    _, _, resi_ndb_seq_can_num = ndb_seq_can_num_dict[(
                        resi_chain, resi_pdb_seq_num, resi_pdb_ins_code, resi_name)]
                    ndb_seq_can_num_to_residue[resi_ndb_seq_can_num] = r
                except:
                    continue
            break

    records = []
    for i, j in combinations(ndb_seq_can_num_to_residue.keys(), 2):
        resi_i = ndb_seq_can_num_to_residue[i]
        resi_j = ndb_seq_can_num_to_residue[j]
        try:
            cb_i = resi_i['CB']
            cb_j = resi_j['CB']
            distance = cb_i - cb_j
            if distance <= interaction_threshold:
                assert i < j
                records.append({
                    "resi_i_ndb_seq_can_num": i,
                    "resi_j_ndb_seq_can_num": j,
                    "distance": cb_i - cb_j
                })
        except:
            continue

    df = pd.DataFrame(records)
    num_all_pairs = len(df)
    x, y = [], []
    for i in range(1, ndb_seq_can_length + 1):
        left_part = df[df['resi_j_ndb_seq_can_num'] <= i]
        right_part = df[df['resi_i_ndb_seq_can_num'] > i]
        diff = num_all_pairs - len(left_part) - len(right_part)
        x.append(i)
        y.append(-diff)

    return dict(zip(x, y))


def assign_domain_for_site(
    domain_boundary_score_dict: dict,
    # ndb seq can num of residue that belongs to this site
    site_nums: List[int],
    domain_length_upper_threshold: int,
    domain_length_lower_threshold: int = 20,
    **kwargs,
) -> tuple:

    def find_domain_boundary(
        interval: tuple,  # both are inclusive
        is_lower_boundary: bool = True,
    ):
        target = interval[0]
        if interval[0] == interval[1]:
            return target

        max_score = -np.inf
        for i in range(interval[0], interval[1] + 1):
            score = domain_boundary_score_dict[i]
            if is_lower_boundary:
                if score >= max_score:
                    target = i
                    max_score = score
            else:
                if score > max_score:
                    target = i
                    max_score = score
        return target

    seq_length = len(domain_boundary_score_dict)
    if seq_length <= domain_length_upper_threshold:
        return 1, seq_length
    min_num = min(site_nums)
    max_num = max(site_nums)
    site_span_length = (max_num - min_num + 1)
    assert site_span_length < domain_length_upper_threshold, f"Site spans too long: {site_nums}"

    radius = (domain_length_upper_threshold - site_span_length) / 2
    lower_interval = (max([1, min_num - int(radius)]), min_num)
    upper_interval = (max_num, min([max_num + math.ceil(radius), seq_length]))
    domain_lower = find_domain_boundary(lower_interval, True)
    domain_upper = find_domain_boundary(upper_interval, False)

    domain_length = domain_upper - domain_lower + 1
    if domain_length < domain_length_lower_threshold:
        logging.warning(f"Domain too short for sites: {site_nums}, {kwargs}")

    return domain_lower, domain_upper  # both are inclusive


def merge_intervals(
    intervals: List[Tuple[int, int]],  # lower -> upper
    length_threshold: int,
    **kwargs,
):
    merged_intervals = []
    lower_sorted_intervals = sorted(intervals, key=lambda x: x[0])
    i = 0
    while True:
        cur_interval = lower_sorted_intervals[i]
        lower_merged_interval, upper_merged_interval = cur_interval

        # after this loop, we define a merged interval
        while True:
            i += 1
            if i < len(lower_sorted_intervals):
                next_interval = lower_sorted_intervals[i]
                lower_next_interval, upper_next_interval = next_interval
                if upper_merged_interval >= lower_next_interval:
                    upper_merged_interval = max(
                        [upper_merged_interval, upper_next_interval])
                    cur_interval = (lower_merged_interval,
                                    upper_merged_interval)
                else:
                    merged_interval_length = upper_merged_interval - lower_merged_interval + 1
                    if merged_interval_length <= length_threshold:
                        merged_intervals.append(cur_interval)
                    else:
                        logging.warning(
                            f"Merged intervals too long for intervals: {intervals}, {kwargs}")
                    break
            else:  # cur interval is the final interval
                merged_interval_length = upper_merged_interval - lower_merged_interval + 1
                if merged_interval_length <= length_threshold:
                    merged_intervals.append(cur_interval)
                else:
                    logging.warning(
                        f"Merged intervals too long for intervals: {intervals}, {kwargs}")
                break
        if i >= len(lower_sorted_intervals):
            break
    return merged_intervals


def run(
    dict_chain_to_seq: dict,
    dict_pdb_file: dict,
    df: pd.DataFrame,
    domain_length_threshold: int = 1700,  # keep 8eey (1601) as one chain
) -> pd.DataFrame:
    site_to_domain = dict()
    for pdb_chain, df_chain in df.groupby(['pdb', 'metal_chain']):

        seq = dict_chain_to_seq[pdb_chain]
        if len(seq) <= domain_length_threshold:
            sites = set(
                zip(df_chain['pdb'], df_chain['metal_chain'], df_chain['metal_pdb_seq_num']))
            for s in sites:
                site_to_domain[s] = ' '
            continue

        pdb, chain = pdb_chain
        pdb_file = dict_pdb_file[pdb]

        # find domains for each site
        tmp_site_to_domain = dict()
        score_dict = get_domain_boundary_score(pdb, pdb_file, chain)
        for metal_pdb_seq_num, df_site in df_chain.groupby(['metal_pdb_seq_num']):
            site_nums = set(df_site['resi_ndb_seq_can_num'])
            domain = assign_domain_for_site(
                score_dict, site_nums, domain_length_threshold, pdb=pdb, chain=chain)
            site = pdb, chain, metal_pdb_seq_num
            tmp_site_to_domain[site] = domain

        # merge domains if there's intersection
        merged_domains = merge_intervals(tmp_site_to_domain.values(
        ), length_threshold=domain_length_threshold, pdb=pdb, chain=chain)
        for k, v in tmp_site_to_domain.items():
            assigned_domain = v
            for m in merged_domains:
                lower_v, upper_v = v
                lower_m, upper_m = m
                if lower_v >= lower_m and upper_v <= upper_m:
                    assigned_domain = m
                    break
            site_to_domain[k] = assigned_domain

    df['domain'] = df.apply(lambda row: site_to_domain[(
        row['pdb'], row['metal_chain'], row['metal_pdb_seq_num'])], axis=1)
    return df


def main(argv):
    FLAGS = flags.FLAGS
    input_pdb = FLAGS.input_pdb
    input_fasta = FLAGS.input_fasta
    input_records = FLAGS.input_records
    output_records = FLAGS.output_records
    output_fasta = FLAGS.output_fasta

    logging.info("Start assigning domains.")
    df_pdb = pd.read_csv(input_pdb, sep="\t")
    dict_pdb_file = dict(zip(df_pdb['pdb'], df_pdb['pdb_file']))
    dict_chain_to_seq = dict()
    for r in SeqIO.parse(input_fasta, "fasta"):
        r: SeqRecord
        dict_chain_to_seq[tuple(r.id.split("_"))] = str(r.seq)
    df_input = pd.read_csv(input_records, sep="\t")

    num_task = mp.cpu_count()
    df_chains = split_data_frame_by_id(
        df_input, id=['pdb', 'metal_chain'], n=num_task)
    jobs = mp.Pool(num_task)
    result = [jobs.apply_async(run, args=(
        dict_chain_to_seq, dict_pdb_file, pg
    )) for pg in df_chains]
    result = [r.get() for r in result]

    df_output = pd.concat(result)
    df_output.to_csv(output_records, sep="\t", index=None)
    seq_records = []
    for pdb_chain_domain, _ in df_output.groupby(['pdb', 'metal_chain', 'domain']):
        pdb, chain, domain = pdb_chain_domain
        chain_seq = dict_chain_to_seq[(pdb, chain)]
        if domain != ' ':
            lower, upper = domain
            seq_records.append(SeqRecord(Seq(
                chain_seq[lower - 1: upper]), id=f"{pdb}_{chain}_{lower}_{upper}", description=""))
        else:
            seq_records.append(
                SeqRecord(Seq(chain_seq), id=f"{pdb}_{chain}", description=""))
    SeqIO.write(seq_records, output_fasta, "fasta")
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
