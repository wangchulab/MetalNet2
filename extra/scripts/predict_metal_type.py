from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from autogluon.tabular import TabularPredictor
from utils.encode_utils import encode


def define_arguments():
    model_path = Path.absolute(
        Path(__file__).parent.parent / "train/MetalNet_metal_type_AutogluonModels")
    flags.DEFINE_string("input_pairs", None, required=True,
                        help="records of pairs, tsv")
    flags.DEFINE_string("output_resi", None,
                        help="metal type pred result at resi level, tsv")
    flags.DEFINE_string("output_pair", None,
                        help="metal type pred result at pair level, tsv")
    flags.DEFINE_string("model_path", str(model_path),
                        help="model for prediction of metal type")
    flags.DEFINE_string("input_encoding_file", None, required=True,
                        help="encoding file, seq_id to file for encoding, tsv")
    flags.DEFINE_string("encode_method", default="esm2",
                        help="encoding method")
    flags.DEFINE_string("encode_strategy", default="avg",
                        help="encoding strategy")


def predict(
    predictor: TabularPredictor,
    df_encoded_pairs: pd.DataFrame,
) -> pd.DataFrame:

    data_x = pd.DataFrame(np.stack(df_encoded_pairs['x']))
    df = df_encoded_pairs.drop(columns=['x']).copy().reset_index()

    df['pred'] = predictor.predict(pd.DataFrame(data_x), as_pandas=False)

    return df


def to_resi_level(df: pd.DataFrame):

    resi_to_metal_type = dict()
    for _, row in df.iterrows():
        pred = row['pred']

        resi_id_1 = row['seq_id'], row['resi_1'], row['resi_seq_posi_1']
        resi_id_2 = row['seq_id'], row['resi_2'], row['resi_seq_posi_2']

        if resi_id_1 in resi_to_metal_type.keys():
            resi_to_metal_type[resi_id_1].add(pred)
        else:
            resi_to_metal_type[resi_id_1] = set([pred])
        if resi_id_2 in resi_to_metal_type.keys():
            resi_to_metal_type[resi_id_2].add(pred)
        else:
            resi_to_metal_type[resi_id_2] = set([pred])

    records = []
    for k, v in resi_to_metal_type.items():
        seq_id, resi, resi_posi = k
        pred = list(v)
        records.append({
            "seq_id": seq_id,
            "resi": resi,
            "resi_seq_posi": resi_posi,
            "pred": ";".join(pred)
        })

    return pd.DataFrame(records)


def main(argv):
    FLAGS = flags.FLAGS
    input_pairs = FLAGS.input_pairs
    input_encoding_file = FLAGS.input_encoding_file
    encode_method = FLAGS.encode_method
    encode_strategy = FLAGS.encode_strategy
    output_resi = FLAGS.output_resi
    output_pair = FLAGS.output_pair
    model_path = FLAGS.model_path

    logging.info(f"Load model from: {model_path}")
    logging.info("Start predicting metal-binding types.")

    df_input_pairs = pd.read_table(input_pairs)
    key = "filter_by_graph"
    if key in df_input_pairs.columns:
        df_input_pairs = df_input_pairs[df_input_pairs[key] == 1]
    df_encoding_file = pd.read_table(input_encoding_file)
    dict_encoding_file = dict(
        zip(df_encoding_file['seq_id'], df_encoding_file['file']))

    df_encoded_pairs = encode(
        df=df_input_pairs,
        dict_file=dict_encoding_file,
        encode_method=encode_method,
        strategy=encode_strategy
    )
    try:
        df_result = predict(
            predictor=TabularPredictor.load(model_path),
            df_encoded_pairs=df_encoded_pairs
        )
    except:
        logging.info("Failed predicting pairs.")
        df_result = pd.DataFrame(columns=df_input_pairs.columns.to_list() + ["pred"])
    if output_resi is not None:
        to_resi_level(df_result).to_csv(output_resi, sep="\t", index=None)
    if output_pair is not None:
        df_input = pd.read_table(input_pairs)
        pd.merge(left=df_result, right=df_input, how='outer').to_csv(output_pair, sep="\t", index=None)
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
