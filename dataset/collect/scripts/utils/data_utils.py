import random
from typing import *

import pandas as pd
from sklearn.model_selection import KFold


def split_data_frame_by_id(
    df: pd.DataFrame,
    id: List[str] = ['pdb'],
    n: int = 5,
    random_state: int = None, 
) -> List[pd.DataFrame]:
    
    # and no id intersection between folds
    columns = sorted(list(set(zip(*[df[c] for c in id])))) # determined order
    l = range(len(columns))
    
    if random_state: kf = KFold(n_splits=n, shuffle=True, random_state=random_state)
    else: kf = KFold(n_splits=n, shuffle=False)
    dfs = []
    for _, valid_index in kf.split(l):
        valid_ids = set([columns[i] for i in valid_index])        
        df_ = df[df.apply(lambda row: tuple([row[c] for c in id]) in valid_ids, axis=1)]
        dfs.append(df_)
    return dfs

def split_train_test(
    df: pd.DataFrame,
    id: List[str] = ['pdb', 'metal_chain'],
    test_size: float = 0.1,
    random_state: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    chains = sorted(list(set(zip(*[df[c] for c in id]))))
    random.seed(random_state)
    random.shuffle(chains)
    test_size = int(len(chains) * test_size)
    test_chains = set(chains[:test_size])

    df_output = df.copy(deep=True)
    df_test = df_output[df_output.apply(lambda row: tuple([row[c] for c in id]) in test_chains, axis=1)]
    df_train = df_output[df_output.apply(lambda row: tuple([row[c] for c in id]) not in test_chains, axis=1)]
    return df_train, df_test