import random
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import KFold

metal_types = {
    'ZN', 'CA', 'MG', 'MN', 'FE',
    'CU', 'NI', 'CO', 'FES', 'SF4',
    'F3S'
}


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


def label_pairs_metal_type(
    df_pairs: pd.DataFrame,  # predicted pairs (as true for metal-binding)
    df_anno: pd.DataFrame,
) -> pd.DataFrame:
    resi_annotated = dict(zip(zip(df_anno['seq_id'], df_anno['resi_domain_posi']), df_anno['metal_resi']))
    records = []
    for _, row in df_pairs.iterrows():

        # predicted as true
        resi_1_id = row['seq_id'], row['resi_seq_posi_1']
        resi_2_id = row['seq_id'], row['resi_seq_posi_2']

        # true predictions
        if resi_1_id in resi_annotated.keys() and resi_2_id in resi_annotated.keys():
            metal_1 = resi_annotated[resi_1_id]
            metal_2 = resi_annotated[resi_2_id]

            # have the same metal types
            if metal_1 == metal_2:
                records.append({
                    "seq_id": row['seq_id'],
                    "resi_seq_posi_1": row['resi_seq_posi_1'],
                    "resi_seq_posi_2": row['resi_seq_posi_2'],
                    "metal_type": metal_1,
                    "label": metal_1
                })

    return pd.DataFrame(records)


def calc_metrics_for_pairs_metal_type(
    df_anno: pd.DataFrame,
    df_pred: pd.DataFrame
) -> dict:

    type_to_anno_resi = dict()
    type_to_pred_resi = dict()
    for m in metal_types:
        type_to_anno_resi[m] = set()
        type_to_pred_resi[m] = set()

    for _, row in df_anno.iterrows():
        metal_type = row['metal_resi']
        resi_id = row['seq_id'], row['resi_domain_posi']
        type_to_anno_resi[metal_type].add(resi_id)

    for _, row in df_pred.iterrows():
        metal_type = row['pred']
        resi_id_1 = row['seq_id'], row['resi_seq_posi_1']
        resi_id_2 = row['seq_id'], row['resi_seq_posi_2']
        # there may be conflicts (r1-r2 -> type1, r1-r3 -> type2), we just put them into different groups
        type_to_pred_resi[metal_type].add(resi_id_1)
        type_to_pred_resi[metal_type].add(resi_id_2)

    result = dict()
    f1_total = 0
    for m in metal_types:
        true_resi = type_to_anno_resi[m]
        pred_resi = type_to_pred_resi[m]
        intersection = true_resi & pred_resi

        m_result = dict()
        recall = len(intersection) / \
            len(true_resi) if len(true_resi) != 0 else 0
        precision = len(intersection) / \
            len(pred_resi) if len(pred_resi) != 0 else 0
        f1 = 2 * recall * precision / \
            (recall + precision) if (recall + precision) != 0 else 0
        m_result['precision'] = precision
        m_result['recall'] = recall
        m_result['f1'] = f1
        result[m] = m_result
        f1_total += f1
    
    result['f1_macro_avg'] = f1_total / len(metal_types)
    
    return result
