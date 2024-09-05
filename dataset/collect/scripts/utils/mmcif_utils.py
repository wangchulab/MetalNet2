from warnings import simplefilter

import pandas as pd
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1, seq3

simplefilter(action="ignore", category=FutureWarning)


def get_ndb_seqs(mmcif_dict: MMCIF2Dict) -> dict:  # pdb_strand_id -> ndb_seq
    ndb_seqs_dict = dict()

    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.pdbx_seq_one_letter_code.html
    # note that:
    # 1. non-canonical residues are represented in the format (***), e.g., 6gez, LVTTL(CRF)VQCF
    # 2. hetero residues are represented as one residue in ndb seq, e.g., 6dtk
    # 3. insertion residue are included (pdb_ins_code), e.g., 1sgt
    seqs = mmcif_dict['_entity_poly.pdbx_seq_one_letter_code']
    chain_ids = mmcif_dict['_entity_poly.pdbx_strand_id']
    for index in range(len(seqs)):
        # homo-multimer sometimes (e.g., "A,B")
        chains = set(chain_ids[index].split(","))
        seq = seqs[index].replace("\n", "")
        for chain in chains:
            ndb_seqs_dict[chain] = seq
    return ndb_seqs_dict


def get_ndb_seqs_can(mmcif_dict: MMCIF2Dict) -> dict:  # pdb_strand_id -> ndb_seq_can
    can_seqs_dict = dict()

    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.pdbx_seq_one_letter_code_can.html
    # ndb_seq_can is used for searching msa, predicting structure
    # note that:
    # 1. seqs with canonical amino acids only (one-letter), it's may be a little bit different
    # compared to the ndb seq when non-canonical amino acids exsits
    seqs = mmcif_dict['_entity_poly.pdbx_seq_one_letter_code_can']
    chain_ids = mmcif_dict['_entity_poly.pdbx_strand_id']
    for index in range(len(seqs)):
        # homo-multimer sometimes (e.g., "A,B")
        chains = set(chain_ids[index].split(","))
        seq = seqs[index].replace("\n", "")
        for chain in chains:
            can_seqs_dict[chain] = seq
    return can_seqs_dict


def get_ndb_seq_three_letter(ndb_seq: str) -> str:
    ndb_seq_three_letter = ""

    seq_len = len(ndb_seq)
    index = 0
    while index < seq_len:
        cur_aa = ""
        cur_char = ndb_seq[index]
        if cur_char == '(':
            index += 1
            cur_char = ndb_seq[index]
            while cur_char != ")":
                cur_aa += cur_char
                index += 1
                cur_char = ndb_seq[index]
        else:
            cur_aa = seq3(cur_char)

        # should be three letter, deoxynucleotides are not allowed in this case (two-letter), e.g., 1f2i
        assert len(cur_aa) == 3
        ndb_seq_three_letter += str.capitalize(cur_aa)
        index += 1
    return ndb_seq_three_letter


def align_two_seq(seq1, seq2) -> dict:
    if seq1 == seq2:
        return dict(zip(range(len(seq1)), range(len(seq2))))

    # align two peptide sequence
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligns = aligner.align(seq1, seq2)
    align = aligns[0].format().strip().split(
        "\n")  # take the first align result
    seq1, seq2 = '', ''
    for i in align:
        if i.startswith("target"):
            try:
                seq1 += i.split()[2]
            except:
                continue  # when the new line has no sequence, e.g., 'target 56'
        elif i.startswith("query"):
            try:
                seq2 += i.split()[2]
            except:
                continue

    # generate path dict: index from seq1 to index from seq2 (matched or replaced)
    seq1_to_seq2 = dict()
    seq1_index, seq2_index = 0, 0
    for i in range(len(seq1)):
        if seq1[i] != "-":
            if seq2[i] != "-":
                seq1_to_seq2[seq1_index] = seq2_index
            seq1_index += 1
        if seq2[i] != "-":
            seq2_index += 1
    return seq1_to_seq2


def get_ndb_seq_can_num(mmcif_dict: MMCIF2Dict) -> dict:

    # (chain, pdb_seq_num, pdb_ins_code, pdb_mon_id) -> (chain, ndb_seq_num, ndb_seq_can_num), structure_id -> sequence_id
    # a one-to-one dict for residue in pdb structure and residue in ndb seq and ndb seq can
    # note that:
    # 1. hetero residues are skipped (only take one of them)
    ndb_seq_can_num_dict = dict()

    ndb_seqs = get_ndb_seqs(mmcif_dict)
    ndb_seqs_can = get_ndb_seqs_can(mmcif_dict)
    ndb_seqs_three_letter = dict()
    align_seqs_to_seqs_can = dict()
    for chain in ndb_seqs.keys():
        try:  # only consider sequence without deoxynucleotides
            ndb_seqs_three_letter[chain] = get_ndb_seq_three_letter(
                ndb_seqs[chain])
            align_seqs_to_seqs_can[chain] = align_two_seq(
                seq1(ndb_seqs_three_letter[chain]), ndb_seqs_can[chain])
        except AssertionError:
            continue

    df = pd.DataFrame()
    # three-letter resi
    df['pdb_mon_id'] = mmcif_dict['_pdbx_poly_seq_scheme.pdb_mon_id']
    df['ndb_seq_num'] = mmcif_dict['_pdbx_poly_seq_scheme.ndb_seq_num']  # starts from 1
    df['pdb_seq_num'] = mmcif_dict['_pdbx_poly_seq_scheme.pdb_seq_num']  # starts from 1
    # pdb chain id
    df['pdb_strand_id'] = mmcif_dict['_pdbx_poly_seq_scheme.pdb_strand_id']
    # {'.', '?', 'A', 'B', ...}
    df['pdb_ins_code'] = mmcif_dict['_pdbx_poly_seq_scheme.pdb_ins_code']

    for chain, df_chain in df.groupby(['pdb_strand_id']):
        if chain not in ndb_seqs_three_letter.keys():
            continue
        ndb_seq_three_letter = ndb_seqs_three_letter[chain]
        align_seq_to_seq_can = align_seqs_to_seqs_can[chain]
        for _, row in df_chain.iterrows():
            pdb_seq_num, pdb_ins_code, pdb_mon_id = row['pdb_seq_num'], row['pdb_ins_code'], row['pdb_mon_id']
            ndb_seq_num = row['ndb_seq_num']
            pdb_seq_num, ndb_seq_num = int(pdb_seq_num), int(ndb_seq_num)
            # unassigned_ins_code_symbols {'.', '?'}
            pdb_ins_code = ' ' if pdb_ins_code in {'.', '?'} else pdb_ins_code

            ndb_seq_three_letter_posi = (ndb_seq_num - 1) * 3
            mon_id = str.upper(ndb_seq_three_letter[ndb_seq_three_letter_posi: (
                ndb_seq_three_letter_posi + 3)])
            if mon_id == pdb_mon_id:  # usually it's equal; if hetero residue exsits, only when pdb resi match ndb_seq resi, e.g., 6dtk in ndb_num 84 and 253
                ndb_seq_posi = ndb_seq_num - 1
                ndb_seq_can_posi = align_seq_to_seq_can[ndb_seq_posi] if ndb_seq_posi in align_seq_to_seq_can.keys(
                ) else -1
                ndb_seq_can_num = ndb_seq_can_posi + 1
                ndb_seq_can_num_dict[(chain, pdb_seq_num, pdb_ins_code, pdb_mon_id)] = (
                    chain, ndb_seq_num, ndb_seq_can_num)
    return ndb_seq_can_num_dict


def get_polymer_id_and_chain_ids(mmcif_dict) -> dict:

    # polymer id is used in https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-<identity>.txt
    # one polymer id points to the same sequence but probably more than 1 chain ids

    polymer_to_chain = dict()
    # some pdbs may not have entity_id, such as 2hya
    polymer_ids = mmcif_dict['_entity_poly.entity_id']
    chain_ids = mmcif_dict['_entity_poly.pdbx_strand_id']
    assert len(polymer_ids) == len(chain_ids)
    for index in range(len(polymer_ids)):
        chains = set(chain_ids[index].split(","))
        poly_id = polymer_ids[index]
        polymer_to_chain[poly_id] = chains
    return polymer_to_chain
