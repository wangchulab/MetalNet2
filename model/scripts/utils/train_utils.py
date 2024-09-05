import random
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import KFold


def split_data_frame_by_id(
    df: pd.DataFrame,
    id: List[str],
    n: int,
    random_state: int = None,
) -> List[pd.DataFrame]:

    # and no id intersection between folds
    columns = sorted(list(set(zip(*[df[c] for c in id]))))  # determined order
    l = range(len(columns))

    if random_state:
        kf = KFold(n_splits=n, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    dfs = []
    for _, valid_index in kf.split(l):
        valid_ids = set([columns[i] for i in valid_index])
        df_ = df[df.apply(lambda row: tuple([row[c]
                          for c in id]) in valid_ids, axis=1)]
        dfs.append(df_)
    return dfs


def split_train_test(
    df: pd.DataFrame,
    id: List[str],
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    chains = sorted(list(set(zip(*[df[c] for c in id]))))
    random.seed(random_state)
    random.shuffle(chains)
    test_size = int(len(chains) * test_size)
    test_chains = set(chains[:test_size])

    df_output = df.copy(deep=True)
    df_test = df_output[df_output.apply(lambda row: tuple(
        [row[c] for c in id]) in test_chains, axis=1)]
    df_train = df_output[df_output.apply(lambda row: tuple(
        [row[c] for c in id]) not in test_chains, axis=1)]
    return df_train, df_test


def label_pairs(
    df_pairs: pd.DataFrame,
    df_anno: pd.DataFrame,
) -> pd.DataFrame:
    resi_annotated = set(zip(df_anno['seq_id'], df_anno['resi_domain_posi']))
    labels = []
    for _, row in df_pairs.iterrows():
        resi_1_id = row['seq_id'], row['resi_seq_posi_1']
        resi_2_id = row['seq_id'], row['resi_seq_posi_2']
        label = -1
        if resi_1_id in resi_annotated and resi_2_id in resi_annotated:
            label = 1
        elif resi_1_id not in resi_annotated and resi_2_id not in resi_annotated:
            label = 0
        labels.append(label)
    df_output = df_pairs.copy()
    df_output['label'] = labels
    df_output = df_output[df_output['label'] != -1]  # a binary problem
    return df_output


def calc_metrics_for_pairs(
    df_anno: pd.DataFrame,
    df_pred: pd.DataFrame
) -> dict:

    true_residues = set(zip(df_anno['seq_id'], df_anno['resi_domain_posi']))
    pred_residues = set()
    for _, row in df_pred.iterrows():
        pred_residues.add((row['seq_id'], row['resi_seq_posi_1']))
        pred_residues.add((row['seq_id'], row['resi_seq_posi_2']))
    intersection = pred_residues & true_residues

    result = dict()
    recall = len(intersection) / \
        len(true_residues) if len(true_residues) != 0 else 0
    precision = len(intersection) / \
        len(pred_residues) if len(pred_residues) != 0 else 0
    f1 = 2 * recall * precision / \
        (recall + precision) if (recall + precision) != 0 else 0
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    return result
