import numpy as np
import pandas as pd
from absl import app, flags, logging
from autogluon.tabular import TabularPredictor
from utils.encode_utils import encode
from utils.io_utils import parse_yaml_file
from utils.train_utils import *


def define_arguments():
    flags.DEFINE_string("input_params", None, required=True,
                        help="params file, yaml")
    flags.DEFINE_string("input_records", None, required=True,
                        help="records of encoding data, csv")
    flags.DEFINE_string("input_anno", None, required=True,
                        help="annotaion file at residue level, tsv")
    flags.DEFINE_string("input_encoding_file", None, required=True,
                        help="encoding file, seq_id to file for encoding, tsv")
    flags.DEFINE_string("output_model", None, required=True, help="model path")
    flags.DEFINE_string("encode_method", None, required=True,
                        help="encoding method to use")
    flags.DEFINE_string("encode_strategy", None,
                        help="encode strategy for the method")


def train(
    df_records: pd.DataFrame,
    dict_encoding_file: dict,
    df_anno: pd.DataFrame,
    encode_method: str,
    encode_strategy: str,
    model_path: str,
    use_balance: bool,
    eval_metric: str,
    random_for_tuning_data: int,
    random_for_balance: int,
):
    logging.debug("Training...")
    logging.info(f"Using params use balance: {use_balance}")
    logging.info(f"Using params eval metric: {eval_metric}")
    logging.info(f"Random num for tuning data: {random_for_tuning_data}")
    logging.info(
        f"Random num for balance pos and neg data: {random_for_balance}")

    # set train, tuning data; balance data(down sampling)
    train_df, tune_df = split_train_test(
        df=label_pairs(df_pairs=df_records, df_anno=df_anno),
        id=['seq_id'],
        test_size=0.1,
        random_state=random_for_tuning_data
    )
    train_pos_df = train_df[train_df['label'] == 1]
    train_neg_df = train_df[train_df['label'] == 0]
    if use_balance:
        train_neg_df = train_neg_df.sample(
            n=len(train_pos_df), random_state=random_for_balance)
    train_df = pd.concat([train_pos_df, train_neg_df])

    # transform to the form that's suitable for autogluon
    train_df = encode(train_df, dict_encoding_file,
                      encode_method, encode_strategy)
    tune_df = encode(tune_df, dict_encoding_file,
                     encode_method, encode_strategy)
    train_data_df = pd.DataFrame(np.stack(train_df['x']))
    tuning_data_df = pd.DataFrame(np.stack(tune_df['x']))
    train_data_df['label'] = train_df['label']
    tuning_data_df['label'] = tune_df['label']
    del train_df, tune_df
    logging.info(f"Training dataset shape (with label): {train_data_df.shape}")
    logging.info(f"Tuning dataset shape (with label): {tuning_data_df.shape}")

    # train
    model = TabularPredictor(
        label="label",
        problem_type='binary',
        eval_metric=eval_metric,
        verbosity=1,
        path=model_path,
    ).fit(
        train_data_df,
        tuning_data=tuning_data_df,
        presets='best_quality',
        use_bag_holdout=True,
    )
    logging.debug("Done.")
    return model


def main(argv):
    FLAGS = flags.FLAGS
    input_params = FLAGS.input_params
    input_encoding_file = FLAGS.input_encoding_file
    input_anno = FLAGS.input_anno
    input_records = FLAGS.input_records
    output_model = FLAGS.output_model
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

    train(
        df_records=df_records,
        df_anno=df_anno,
        dict_encoding_file=dict_encoding_file,
        encode_method=encode_method,
        encode_strategy=encode_strategy,
        use_balance=hparams.use_balance,
        eval_metric=hparams.eval_metric,
        random_for_balance=hparams.random_balance,
        random_for_tuning_data=hparams.random_tuning,
        model_path=output_model
    )


if __name__ == "__main__":
    define_arguments()
    app.run(main)
