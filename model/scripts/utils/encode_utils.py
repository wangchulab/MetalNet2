from abc import ABC, abstractmethod
from typing import Union
from warnings import simplefilter

import numpy as np
import pandas as pd

simplefilter(action="ignore", category=FutureWarning)


def encode(
    df: pd.DataFrame,  # coevo pair
    dict_file: dict,
    encode_method: str,
    strategy: str,
    drop_none: bool = True,
) -> pd.DataFrame:
    records = []
    for seq_id, df_seq in df.groupby(['seq_id']):
        encoder: PairEncoder = get_encoder(encode_method)
        file_for_encoding = dict_file[seq_id]
        encoder.parse(file_for_encoding, strategy=strategy, seq_id=seq_id)
        for _, row in df_seq.iterrows():
            posi_1 = row['resi_seq_posi_1']
            posi_2 = row['resi_seq_posi_2']
            try:
                encoding = encoder.encode(
                    posi_1=posi_1, posi_2=posi_2, strategy=strategy)
            except:
                encoding = None  # can't extract encoding, such as NeighborStatisticEncoder
                if drop_none:
                    continue
            records.append({"x": encoding, **row})
    return pd.DataFrame(records)


def get_encoder(method):
    options = {
        "freq_mtx",
        "esm2",
        "af2_pair_repr",
        "af2_resi_repr",
        "nb"  # neighbor
    }
    if method not in options:
        raise ValueError

    encoder = None
    if method == "freq_mtx":
        encoder = MutationFrequencyEncoder()
    elif method == "esm2":
        encoder = ESMEncoder()
    elif method == "af2_pair_repr":
        encoder = AF2PairEncoder()
    elif method == "af2_resi_repr":
        encoder = AF2ResiEncoder()
    elif method == "nb":
        encoder = NeighborStatisticEncoder()
    else:
        raise NotImplementedError

    return encoder


def convert_single_to_pair(
    posi_1: int,
    posi_2: int,
    encoding_1: np.ndarray,
    encoding_2: np.ndarray,
    strategy: str,
):
    options = {"max", "avg", "cat"}
    if strategy not in options:
        raise ValueError

    encoding = None
    if strategy == "max":
        encoding = np.maximum(encoding_1, encoding_2)
    elif strategy == "avg":
        encoding = np.average([encoding_1, encoding_2], axis=0)
    elif strategy == "cat":
        encoding = np.concatenate([encoding_1, encoding_2], axis=0) if posi_1 < posi_2 \
            else np.concatenate([encoding_2, encoding_1], axis=0)

    return encoding


class PairEncoder(ABC):

    @abstractmethod
    def parse(self, file: str, **kwargs):
        pass

    @abstractmethod
    def encode(self, posi_1: int, posi_2: int, **kwargs) -> np.ndarray:
        pass


class MutationFrequencyEncoder(PairEncoder):

    aa_msa = [
        'C', 'H', 'D', 'E', 'N',
        'S', 'T', 'K', 'G', 'Q',
        'Y', 'L', 'A', 'V', 'R',
        'I', 'M', 'F', 'W', 'P',
        '-'
    ]
    aa_msa_to_index = dict(zip(aa_msa, [i for i in range(len(aa_msa))]))

    def parse(self, file: str, **kwargs):
        self.aln_mtx = self.get_aln_mtx(file)

    def encode(self, posi_1: int, posi_2: int, **kwargs) -> np.ndarray:

        aln_mtx = self.aln_mtx

        fq_mtx = np.zeros((len(self.aa_msa), len(self.aa_msa)))
        for r in range(aln_mtx.shape[0]):
            aa1_idx = aln_mtx[r][posi_1]
            aa2_idx = aln_mtx[r][posi_2]
            fq_mtx[aa1_idx, aa2_idx] += 1
        fq_mtx = fq_mtx / np.sum(fq_mtx)

        return fq_mtx.flatten()

    def get_aln_mtx(self, msa_file: str) -> np.ndarray:
        from .io_utils import parse_a3m_file
        aa_msa_to_index = self.aa_msa_to_index
        seqs = parse_a3m_file(msa_file)
        mtx = []
        for seq in seqs:
            mtx.append([aa_msa_to_index[aa] if aa in aa_msa_to_index.keys(
            ) else aa_msa_to_index['-'] for aa in seq])
        return np.array(mtx)


class ESMEncoder(PairEncoder):

    def parse(self, file: str, **kwargs):
        import torch
        self.encoding = torch.load(file)['representations'][33]

    def encode(self, posi_1: int, posi_2: int, **kwargs) -> np.ndarray:
        import torch
        esm_mtx: torch.Tensor = self.encoding

        strategy = kwargs['strategy']
        posi_1_encoding = esm_mtx[posi_1].numpy()
        posi_2_encoding = esm_mtx[posi_2].numpy()
        encoding = convert_single_to_pair(
            posi_1, posi_2, posi_1_encoding, posi_2_encoding, strategy)
        return encoding


class AF2PairEncoder(PairEncoder):

    def parse(self, file: str, **kwargs):
        self.pair_repr = np.load(file)

    def encode(self, posi_1: int, posi_2: int, **kwargs) -> np.ndarray:
        encoding = self.pair_repr[posi_1][posi_2].copy()
        return encoding.astype(np.float32)


class AF2ResiEncoder(PairEncoder):

    def parse(self, file: str, **kwargs):
        self.resi_repr = np.load(file)

    def encode(self, posi_1: int, posi_2: int, **kwargs) -> np.ndarray:
        af2_resi_mtx = self.resi_repr

        strategy = kwargs['strategy']
        posi_1_encoding = af2_resi_mtx[posi_1]
        posi_2_encoding = af2_resi_mtx[posi_2]
        encoding = convert_single_to_pair(
            posi_1, posi_2, posi_1_encoding, posi_2_encoding, strategy)
        return encoding.astype(np.float32)


class NeighborStatisticEncoder(PairEncoder):

    from typing import List

    from Bio.PDB.Chain import Chain
    from Bio.PDB.NeighborSearch import NeighborSearch
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    common_amino_acids = {
        'Ala', 'Cys', 'Asp', 'Glu', 'Phe',
        'Gly', 'His', 'Ile', 'Lys', 'Leu',
        'Met', 'Asn', 'Pro', 'Gln', 'Arg',
        'Ser', 'Thr', 'Val', 'Trp', 'Tyr',
    }
    layers = [4, 6, 8, 10]

    def parse(self, file: str, **kwargs):

        from Bio.PDB.NeighborSearch import NeighborSearch

        from .io_utils import parse_mmcif_file

        seq_id = kwargs['seq_id']
        struct, mmcif_dict = parse_mmcif_file(seq_id, file)
        _, chain_id, domain = NeighborStatisticEncoder.seq_id_to_domain_record(
            seq_id)
        residues = NeighborStatisticEncoder.residues_in_domain(
            struct, mmcif_dict, chain_id, domain)

        self.seq_posi_to_residue = dict()
        atoms = []
        for r in residues:
            self.seq_posi_to_residue[r.id[-1]] = r
            atoms.extend(list(r.get_atoms()))
        self.ns = NeighborSearch(atoms)

    def encode(self, posi_1: int, posi_2: int, **kwargs) -> np.ndarray:
        strategy = kwargs['strategy']
        encoding = None
        ns = self.ns
        layers = self.layers

        resi_1 = self.seq_posi_to_residue[posi_1]
        resi_2 = self.seq_posi_to_residue[posi_2]

        if strategy == "centroid":
            centroid = NeighborStatisticEncoder.get_centroid([resi_1, resi_2])
            encoding = NeighborStatisticEncoder.encode_resi_by_num_of_neighbors(
                ns, centroid, layers)
        else:
            x1 = NeighborStatisticEncoder.encode_resi_by_num_of_neighbors(
                ns, resi_1["CA"].get_vector().get_array(), layers)
            x2 = NeighborStatisticEncoder.encode_resi_by_num_of_neighbors(
                ns, resi_2["CA"].get_vector().get_array(), layers)
            encoding = convert_single_to_pair(posi_1, posi_2, x1, x2, strategy)

        return encoding

    def residues_in_domain(
        struct: Structure,
        mmcif_dict: dict,
        chain_id: str,
        domain: Union[tuple, str],
    ):
        from .mmcif_utils import get_ndb_seq_can_num

        seq_can_num_lower = 1 if domain == " " else domain[0]
        seq_can_num_upper = np.inf if domain == " " else domain[1]

        resi_pdb_id_to_resi_seq_id = get_ndb_seq_can_num(
            mmcif_dict)  # struct id -> chain id
        target_chain = None
        for c in struct.get_chains():
            if c.get_id() == chain_id:
                target_chain = c
                break
        residues = []
        for r in target_chain.get_residues():
            key = (chain_id, r.id[1], r.id[2], r.get_resname())
            try:
                _, _, ndb_seq_can_num = resi_pdb_id_to_resi_seq_id[key]
                if ndb_seq_can_num > seq_can_num_upper or ndb_seq_can_num < seq_can_num_lower:
                    continue
                fasta_posi = ndb_seq_can_num - seq_can_num_lower
            except KeyError:
                continue
            # NOTE: append posi in the whole seq (starts from 0), this is not compatible with __repr__ in Residue class
            r.id = r.id + (fasta_posi,)
            residues.append(r)
        return residues

    def seq_id_to_domain_record(seq_id: str) -> tuple:
        # id -> (pdb, chain, domain)
        # domain: resi_ndb_seq_can (start, end) both inclusive
        try:
            pdb, chain, start, end = seq_id.split("_")
            domain = (int(start), int(end))
        except:
            pdb, chain = seq_id.split("_")
            domain = " "
        return (pdb, chain, domain)

    def encode_resi_by_num_of_neighbors(
        ns: NeighborSearch,  # define the search space
        source_coordinate: np.ndarray,
        layers: List[int]
    ) -> np.ndarray:

        layers.sort()
        radius = layers[-1]
        neighbors = ns.search(source_coordinate, radius, level="R")

        layer_to_resi = dict()
        for l in layers:
            layer_to_resi[l] = []

        for r in neighbors:
            d = NeighborStatisticEncoder.calc_resi_distance(
                r, source_coordinate)
            target_layer = None
            for l in layers:
                if d <= l:
                    target_layer = l
                    break
            layer_to_resi[target_layer].append(r)

        encodings = []
        common_amino_acids = NeighborStatisticEncoder.common_amino_acids
        for l in layers:
            residues_in_layer = layer_to_resi[l]
            aa_count = dict(
                zip(common_amino_acids, [0 for _ in range(len(common_amino_acids))]))
            for r in residues_in_layer:
                aa = str.capitalize(r.get_resname())
                if aa in common_amino_acids:
                    aa_count[aa] += 1
            encoding = np.array(list(aa_count.values()))
            encodings.append(encoding)
        return np.concatenate(encodings, axis=0)

    def calc_resi_distance(
        resi: Residue,
        source_coordinate: np.ndarray,
    ):
        distances = []
        for a in resi.get_atoms():
            distances.append(np.linalg.norm(
                a.get_vector().get_array() - source_coordinate, 2))
        return np.min(distances)

    def get_centroid(residues: List[Residue]):
        from itertools import product

        residue_to_atoms = dict()
        for r in residues:
            residue_to_atoms[r] = [a for a in r.get_atoms()]

        min_distance = np.inf
        centroid = None
        # each residue gives one hetero atom
        for i in product(*residue_to_atoms.values()):

            atoms_coordinates = []
            for a in i:
                atoms_coordinates.append(a.get_vector().get_array())

            cur_centroid = np.mean(np.array(atoms_coordinates), axis=0)
            distance_sum = np.sum(
                [np.linalg.norm(a.get_vector().get_array() - cur_centroid, 2) for a in i])
            if distance_sum <= min_distance:
                min_distance = distance_sum
                centroid = cur_centroid

        return centroid
