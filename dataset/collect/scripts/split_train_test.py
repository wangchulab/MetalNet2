import pandas as pd
from absl import app, flags, logging
from utils.data_utils import split_train_test


def define_arguments():
    flags.DEFINE_string("input_records", None, required=True, help="metal binding chains file, csv")
    flags.DEFINE_string("output_records", None, required=True, help="metal binding chains file with data type, csv")
    flags.DEFINE_string("output_train_records", None, help="train data records")
    flags.DEFINE_string("output_test_records", None, help="test data records")
    
def main(argv):
    FLAGS = flags.FLAGS
    input_records = FLAGS.input_records
    output_records = FLAGS.output_records
    output_train_records = FLAGS.output_train_records
    output_test_records = FLAGS.output_test_records

    logging.info("Splitting dataset into train dataset and test dataset.")
    df_input = pd.read_csv(input_records, sep="\t")
    df_train, df_test = split_train_test(df_input, random_state=10)
    df_train['data_type'] = "train_data"
    df_test['data_type'] = "test_data"
    df_output = pd.concat([df_train, df_test])
    df_output.to_csv(output_records, sep="\t", index=None)
    if output_train_records: df_train.to_csv(output_train_records, sep="\t", index=None)
    if output_test_records: df_test.to_csv(output_test_records, sep="\t", index=None)
    logging.info("Done.")


if __name__ == "__main__":
    define_arguments()
    app.run(main)