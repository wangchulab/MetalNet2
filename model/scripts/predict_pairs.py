import os.path as osp

import numpy as np
import pandas as pd
from absl import app, flags, logging
from autogluon.tabular import TabularPredictor
from graphviz import Graph
from utils.encode_utils import encode
from utils.graph_filter import filter_pairs_by_graph

ched_color = {
    'C': "#F8E695",
    'H': "#6CA7DB",
    'E': "#D6A6CA",
    'D': "#BAA0CA"
}


def define_arguments():
    script_dir = osp.dirname(osp.abspath(__file__))
    model_dir = osp.abspath(
        osp.join(script_dir, "../train/MetalNet_AutogluonModels/"))
    flags.DEFINE_string("input_encoding_file", None, required=True,
                        help="encoding file, seq_id to file for encoding, tsv")
    flags.DEFINE_string("input_pairs", None, required=True,
                        help="records of pairs, csv")
    flags.DEFINE_string("output_result", None, required=True,
                        help="output pair prob file, csv")
    flags.DEFINE_string("encode_method", "esm2",
                        help="encoding method to use")
    flags.DEFINE_string("encode_strategy", "avg",
                        help="encode strategy for the method")
    flags.DEFINE_string("model_path", model_dir,
                        help="path to autogluon model")
    flags.DEFINE_float("prob_threshold", 0.8,
                       help="probability to define a metal-binding pair")
    flags.DEFINE_integer("node_num_threshold", 4,
                         help="graph nodes num to define a metal net")
    flags.DEFINE_bool("to_image", False, help="export pair graph to file")


def export_graph(
    name: str,
    output_dir: str,
    df_pairs: pd.DataFrame,  # ched positive pairs for each seq
):
    dot = Graph(comment=name, format="pdf", directory=output_dir)
    for _, row in df_pairs.iterrows():
        resi_1 = row['resi_1']
        resi_2 = row['resi_2']
        seq_num_1 = row['resi_seq_posi_1'] + 1
        seq_num_2 = row['resi_seq_posi_2'] + 1
        tag_1 = resi_1 + str(seq_num_1)
        tag_2 = resi_2 + str(seq_num_2)
        dot.node(tag_1, style="radial",
                 fillcolor=ched_color[resi_1], penwidth='1.5', fontname="Arial")
        dot.node(tag_2, style="radial",
                 fillcolor=ched_color[resi_2], penwidth='1.5', fontname="Arial")
        dot.edge(tag_1, tag_2, penwidth='1.5')
    dot.render(f"{name}.gv", view=False)


def predict(
    predictor: TabularPredictor,
    df_encoded_pairs: pd.DataFrame,
    prob_threshold: float,
    node_num_threshold: int,
) -> pd.DataFrame:

    data_x = pd.DataFrame(np.stack(df_encoded_pairs['x']))
    df = df_encoded_pairs.drop(columns=['x']).copy().reset_index()

    # predict by autogluon: with column 'prob' for the probability of a metal-binding pair
    df['prob'] = pd.Series(predictor.predict_proba(
        pd.DataFrame(data_x), as_pandas=False)[:, 1])
    # filter by graph: with column 'filter_by_graph' for negative (0) and positive (1)
    df = filter_pairs_by_graph(
        df_pairs=df,
        prob_threshold=prob_threshold,
        node_num_threshold=node_num_threshold
    )

    return df


def main(argv):
    FLAGS = flags.FLAGS
    input_encoding_file = FLAGS.input_encoding_file
    input_pairs = FLAGS.input_pairs
    output_result = FLAGS.output_result

    model_path = FLAGS.model_path
    encode_method = FLAGS.encode_method
    encode_strategy = FLAGS.encode_strategy
    prob_threshold = FLAGS.prob_threshold
    node_num_threshold = FLAGS.node_num_threshold
    to_image = FLAGS.to_image

    logging.info(f"Load model from: {model_path}")
    logging.info(f"Using params prob threshold: {prob_threshold}")
    logging.info(
        f"Using params node num threshold in graph filter: {node_num_threshold}")
    logging.info(f"Using encoding method: {encode_method}")
    logging.info(f"Using encoding strategy: {encode_strategy}")

    logging.info("Loading model and data...")
    model = TabularPredictor.load(model_path)
    df_input_pairs = pd.read_table(input_pairs)
    df_encoding_file = pd.read_table(input_encoding_file)
    dict_encoding_file = dict(
        zip(df_encoding_file['seq_id'], df_encoding_file['file']))
    logging.info("Done.")

    logging.info("Start encoding and predicting metal-binding pairs.")
    df_encoded_pairs = encode(
        df=df_input_pairs,
        dict_file=dict_encoding_file,
        encode_method=encode_method,
        strategy=encode_strategy,
    )
    try:
        df_output = predict(
            predictor=model,
            df_encoded_pairs=df_encoded_pairs,
            prob_threshold=prob_threshold,
            node_num_threshold=node_num_threshold
        )
    except:
        logging.info("Failed predicting pairs.")
        df_output = pd.DataFrame(columns=df_input_pairs.columns.to_list() + ["prob", "filter_by_graph"])
    df_output.to_csv(output_result, sep="\t", index=None)
    if to_image:
        output_dir = osp.dirname(osp.abspath(output_result))
        for seq_id, df_seq in df_output.groupby(["seq_id"]):
            # labeled as 1 after graph filter
            df_seq = df_seq[df_seq['filter_by_graph'] == 1]
            export_graph(seq_id, output_dir, df_seq)
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)
