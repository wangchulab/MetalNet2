from abc import ABC, abstractmethod
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
        "esm2",
    }
    if method not in options:
        raise ValueError

    encoder = None
    if method == "esm2":
        encoder = ESMEncoder()
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
