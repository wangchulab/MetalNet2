from itertools import product
from typing import Union

import numpy as np
import pandas as pd
from absl import app, flags, logging
from autogluon.tabular import TabularPredictor
from utils.encode_utils import encode
from utils.graph_filter import filter_pairs_by_cys_pair, filter_pairs_by_graph
from utils.io_utils import parse_yaml_file
from utils.train_utils import *


def define_arguments():
    flags.DEFINE_string("input_params", None, required=True,
                        help="params file for train, yaml")
    flags.DEFINE_string("input_records", None, required=True,
                        help="coevo pairs file, tsv")
    flags.DEFINE_string("input_anno", None, required=True,
                        help="annotaion file at residue level, tsv")
    flags.DEFINE_string("input_encoding_file", None, required=True,
                        help="encoding file, seq_id to file for encoding, tsv")
    flags.DEFINE_string("output_result", None, required=True,
                        help="tuning result file, tsv")
    flags.DEFINE_string("encode_method", None, required=True,
                        help="encoding method to use")
    flags.DEFINE_string("encode_strategy", None,
                        help="encode strategy for the method")


def tune_by_cross_validation(
    df_records: pd.DataFrame,
    df_anno: pd.DataFrame,
    dict_encoding_file: dict,
    encode_method: str,
    encode_strategy: str,
    use_balance: bool,
    eval_metric: str,
    random_for_folds: int,
    randoms_for_tuning_data: List[int],
    randoms_for_balance: List[int],
    num_folds: int = 5,
) -> List[pd.DataFrame]:

    logging.debug("Tuning...")
    logging.info(f"Using encoding method: {encode_method}")
    logging.info(f"Using encoding strategy: {encode_strategy}")
    logging.info(f"Using params use balance: {use_balance}")
    logging.info(f"Using params eval metric: {eval_metric}")
    logging.info(f"Using params num cross-validation folds: {num_folds}")
    logging.info(f"Random num for cross valid folds: {random_for_folds}")
    logging.info(f"Random nums for tuning data: {randoms_for_tuning_data}")
    logging.info(
        f"Random nums for balance pos and neg data: {randoms_for_balance}")

    folds_dfs = split_data_frame_by_id(
        df_records, id=['seq_id'], n=num_folds, random_state=random_for_folds)
    valid_result_dfs = []
    for i in range(num_folds):

        logging.debug(f"Cross validation on fold {i}...")

        # set data
        # set up train, valid (coevo pairs)
        logging.debug(f"Setting up data...")
        valid_df = folds_dfs.pop(0)
        train_df = pd.concat(folds_dfs, copy=True)
        folds_dfs.append(valid_df)
        valid_df = valid_df.copy()

        # set up train, tune
        train_df, tune_df = split_train_test(
            # labelled coevo pairs (add 'label' column)
            df=label_pairs(df_pairs=train_df, df_anno=df_anno),
            id=['seq_id'],
            test_size=0.1,  # tuning data, 10 % of training data
            random_state=randoms_for_tuning_data[i]
        )
        train_pos_df = train_df[train_df['label'] == 1]
        train_neg_df = train_df[train_df['label'] == 0]
        if use_balance:  # balance train data (down sampling)
            train_neg_df = train_neg_df.sample(
                n=len(train_pos_df), random_state=randoms_for_balance[i])
        train_df = pd.concat([train_pos_df, train_neg_df])

        # encode data
        train_df = encode(train_df, dict_encoding_file,
                          encode_method, encode_strategy)
        tune_df = encode(tune_df, dict_encoding_file,
                         encode_method, encode_strategy)
        train_data_df = pd.DataFrame(np.stack(train_df['x']))
        tuning_data_df = pd.DataFrame(np.stack(tune_df['x']))
        train_data_df['label'] = train_df['label']
        tuning_data_df['label'] = tune_df['label']
        del train_df, tune_df
        logging.info(
            f"Training dataset shape (with label): {train_data_df.shape}")
        logging.info(
            f"Tuning dataset shape (with label): {tuning_data_df.shape}")
        logging.info("Done.")

        # train model
        logging.debug(f"Training fold {i}...")
        predictor = TabularPredictor(
            label="label",
            problem_type='binary',
            eval_metric=eval_metric,
            verbosity=1,
        ).fit(
            train_data_df,
            tuning_data=tuning_data_df,
            presets='best_quality',
            use_bag_holdout=True,
        )
        logging.debug("Done.")

        # predict on validation fold
        logging.debug(f"Validating...")
        valid_df = encode(valid_df, dict_encoding_file,
                          encode_method, encode_strategy)
        valid_data_df = pd.DataFrame(np.stack(valid_df['x']))
        logging.info(f"Validation dataset shape: {valid_data_df.shape}")

        prob = predictor.predict_proba(valid_data_df, as_pandas=False)
        valid_result_df = valid_df.drop(columns=['x']).copy().reset_index()
        valid_result_df['prob'] = pd.Series(prob[:, 1])
        valid_result_dfs.append(valid_result_df)
        logging.debug("Done.")
    return valid_result_dfs


def filter_cross_validation_results(
    valid_result_dfs: List[pd.DataFrame],
    df_anno: pd.DataFrame,
    prob_thresholds: Union[float, List[float]],
    node_num_thresholds: Union[int, List[int]],
) -> pd.DataFrame:

    logging.debug("Filtering...")
    logging.info(
        f"Using params node num threshold in graph filter: {node_num_thresholds}")
    logging.info(f"Using params prob threshold: {prob_thresholds}")

    results = []
    valid_result_all_df = pd.concat(valid_result_dfs)

    if not isinstance(prob_thresholds, list):
        prob_thresholds = [prob_thresholds]
    if not isinstance(node_num_thresholds, list):
        node_num_thresholds = [node_num_thresholds]
    for prob_threshold, node_num_threshold in product(prob_thresholds, node_num_thresholds):

        valid_result_all_df = filter_pairs_by_graph(
            valid_result_all_df, prob_threshold, node_num_threshold)  # add 'filter_by_graph' as one column
        valid_result_all_df = filter_pairs_by_cys_pair(
            valid_result_all_df)  # add 'filter_by_cys_pair' as one column

        # without graph filter
        result = calc_metrics_for_pairs(
            df_anno, valid_result_all_df[valid_result_all_df['prob'] >= prob_threshold])
        result.update({
            'filter_by_graph': False,
            'filter_by_cys_pair': False,
            'prob_threshold': prob_threshold,
            'node_num_threshold': node_num_threshold,
        })
        results.append(result)

        # with graph filter
        result = calc_metrics_for_pairs(
            df_anno, valid_result_all_df[valid_result_all_df['filter_by_graph'] == 1])  # without cys pair filter
        result.update({
            'filter_by_graph': True,
            'filter_by_cys_pair': False,
            'prob_threshold': prob_threshold,
            'node_num_threshold': node_num_threshold,
        })
        results.append(result)
        result = calc_metrics_for_pairs(
            df_anno, valid_result_all_df[valid_result_all_df['filter_by_cys_pair'] == 1])  # with cys pair filter
        result.update({
            'filter_by_graph': True,
            'filter_by_cys_pair': True,
            'prob_threshold': prob_threshold,
            'node_num_threshold': node_num_threshold,
        })
        results.append(result)
    logging.debug("Done.")
    return pd.DataFrame(results)


def main(argv):
    FLAGS = flags.FLAGS
    input_params = FLAGS.input_params
    input_encoding_file = FLAGS.input_encoding_file
    input_records = FLAGS.input_records
    input_anno = FLAGS.input_anno
    output_result = FLAGS.output_result
    encode_method = FLAGS.encode_method
    encode_strategy = FLAGS.encode_strategy

    # init data
    logging.set_verbosity('debug')
    logging.debug("Loading input files...")
    hparams = parse_yaml_file(input_params)
    df_records = pd.read_table(input_records)
    df_anno = pd.read_table(input_anno)
    df_encoding_file = pd.read_table(input_encoding_file)
    dict_encoding_file = dict(
        zip(df_encoding_file['seq_id'], df_encoding_file['file']))
    logging.debug("Done.")

    train_metric_dfs = []
    for i in range(hparams.num_repeats):
        logging.info(f"Start training with repeat No.{i}...")
        valid_result_dfs = tune_by_cross_validation(
            df_records=df_records,
            df_anno=df_anno,  # use for label df_records (binary clf task)
            dict_encoding_file=dict_encoding_file,
            encode_method=encode_method,
            encode_strategy=encode_strategy,
            use_balance=hparams.use_balance,
            eval_metric=hparams.eval_metric,
            random_for_folds=hparams.random_valid[i],
            randoms_for_balance=hparams.randoms_balance[i],
            randoms_for_tuning_data=hparams.randoms_tuning[i]
        )
        eval_result = filter_cross_validation_results(
            valid_result_dfs,
            df_anno=df_anno,
            prob_thresholds=hparams.prob_threshold,
            node_num_thresholds=hparams.node_num_threshold
        )
        eval_result['repeat'] = i
        train_metric_dfs.append(eval_result)
        logging.info(f"Finish repeat No.{i}.")
    pd.concat(train_metric_dfs).to_csv(output_result, sep="\t", index=None)
    logging.debug("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
