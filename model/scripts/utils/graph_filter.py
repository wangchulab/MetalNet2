from warnings import simplefilter

import networkx as nx
import pandas as pd

simplefilter(action="ignore", category=FutureWarning)


def find_ring(g: nx.Graph) -> nx.Graph:
    ''' Remove leaf nodes from a graph '''
    g_ = g.copy()
    while True:
        leaf_nodes = [n for n, degree in g_.degree() if degree < 2]
        if len(leaf_nodes) == 0:
            break
        else:
            g_.remove_nodes_from(leaf_nodes)
    return g_


def assign_label_by_graph(
    pairs: dict,  # edge (posi_1: int, posi_2: int) -> pred_prob
    num_threshold: int,
) -> dict:  # edge -> {0, 1}

    # build graph
    g = nx.Graph()
    g.add_edges_from(pairs.keys(), prob=pairs.values())

    # remove leaf nodes and corresponding edges
    g = find_ring(g)
    keys_sorted = set([tuple(sorted(key)) for key in pairs.keys()])
    edges_sorted = set([tuple(sorted(edge)) for edge in g.edges])
    pairs_preserved = edges_sorted & keys_sorted

    # check if use rescue: when num of nodes after removing leaf nodes
    # is less than threshold, trigger rescue mechanism.
    # NOTE: when g has more than one connected subgraph, the num of nodes
    # NOTE: is the sum of nodes of all connected subgraphs.
    pairs_final = pairs_preserved
    if g.number_of_nodes() < num_threshold:

        pairs_removed = set(pairs.keys()) - pairs_preserved
        if len(pairs_removed) > 0:

            # here we set the rescue num of pairs no more than num_threshold
            pairs_rescued = sorted(pairs_removed, key=lambda x: pairs[x], reverse=True)[
                :min(num_threshold, len(pairs_removed))]
            pairs_final |= set(pairs_rescued)

    result = dict()
    for p in pairs.keys():
        # positive vs negtive pred result
        result[p] = 1 if p in pairs_final else 0
    return result


def filter_pairs_by_graph(
    df_pairs: pd.DataFrame,
    prob_threshold: float,
    node_num_threshold: int,
) -> pd.DataFrame:

    pred_label_dict = dict()
    for seq_id, df in df_pairs.groupby(['seq_id']):
        pairs_to_prob = dict()
        for _, row in df.iterrows():
            pair = (row['resi_seq_posi_1'], row['resi_seq_posi_2'])
            if row['prob'] < prob_threshold:
                pred_label_dict[(seq_id, *pair)] = 0  # negtive
            else:
                pairs_to_prob[pair] = row['prob']  # need to determine by graph
        pairs_to_label = assign_label_by_graph(
            pairs_to_prob, node_num_threshold)
        for pair, label in pairs_to_label.items():
            pred_label_dict[(seq_id, *pair)] = label

    df_pairs['filter_by_graph'] = df_pairs.apply(lambda row: pred_label_dict[(
        row['seq_id'], row['resi_seq_posi_1'], row['resi_seq_posi_2'])], axis=1)
    return df_pairs


def filter_pairs_by_cys_pair(df_pairs: pd.DataFrame) -> pd.DataFrame:
    # remove cys-cys pair (cluster size: 2), df_pairs should be filtered by graph ahead

    df_filtered_by_graph = df_pairs[df_pairs['filter_by_graph'] == 1]
    dropped_cys_residues = set()
    for seq_id, df in df_filtered_by_graph.groupby(['seq_id']):
        g = nx.Graph()
        for _, row in df.iterrows():
            resi1_id = (seq_id, row['resi_seq_posi_1'], row['resi_1'])
            resi2_id = (seq_id, row['resi_seq_posi_2'], row['resi_2'])
            g.add_edge(resi1_id, resi2_id)

        for sub_g in nx.connected_components(g):
            if len(sub_g) == 2:  # the graph only coontains a single pair
                resi1_id, resi2_id = tuple(sub_g)
                _, _, resi1 = resi1_id
                _, _, resi2 = resi2_id
                if resi1 == "C" and resi2 == "C":  # probably ssbond:
                    dropped_cys_residues.add(resi1_id)
                    dropped_cys_residues.add(resi2_id)
    # a retained pair is a single cyc-cys if and only if one of them in dropped cys residues
    df_pairs['filter_by_cys_pair'] = df_pairs.apply(lambda row:
                                                    1 if row['filter_by_graph'] == 1 and (row['seq_id'], row['resi_seq_posi_1'], row['resi_1']) not in dropped_cys_residues
                                                    else 0,
                                                    axis=1)
    return df_pairs
