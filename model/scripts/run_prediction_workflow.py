import os
import os.path as osp

from absl import app, flags


def define_arguments():
    flags.DEFINE_string("input_fasta", None, help="fasta")
    flags.DEFINE_string("input_msa", None, help="seq_id to msa_file, tsv")
    flags.DEFINE_string("msa_source", "user_defined",
                        help="two types: user_defined, colabfold.")
    flags.DEFINE_string("output_pairs", None, required=True,
                        help="output predicted metal-binding pairs, tsv")
    flags.DEFINE_bool("keep_inter_files", False,
                      help="do not remove intermediate files")
    flags.DEFINE_bool("to_image", False, help="export pair graph to file")
    flags.DEFINE_integer("cuda", None, help="the gpu device")
    flags.DEFINE_bool("metal_type", False, help="pred metal type")


def run(
    input_fasta: str,
    input_msa: str,
    msa_source: str,
    output_pairs: str,
    keep_inter_files: bool,
    to_image: bool,
    cuda: int,
    metal_type: bool
):
    scripts_dir = osp.dirname(osp.abspath(__file__))
    search_msa_script = osp.join(scripts_dir, "search_msa.py")
    extract_coevo_pairs_script = osp.join(
        scripts_dir, "extract_coevo_pairs.py")
    extract_esm2_script = osp.join(scripts_dir, "extract_esm2.py")
    predict_script = osp.join(scripts_dir, "predict_pairs.py")
    predict_metal_type_script = osp.join(scripts_dir, "../../extra/scripts/predict_metal_type.py")

    work_dir = osp.dirname(osp.abspath(output_pairs))
    coevo_file = osp.join(work_dir, "coevo.csv")
    esm2_file = osp.join(work_dir, "esm_files.tsv")
    esm2_dir = osp.join(work_dir, "esm2/")

    # search msa
    if msa_source == "colabfold":
        msa_dir = osp.join(work_dir, "msa/")
        input_msa = osp.join(work_dir, "msa_files.tsv")
        assert not os.system(f"\
            python {search_msa_script} \
            --input_fasta {input_fasta} \
            --output_dir {msa_dir} \
            --output_msa {input_msa} \
        ")

    # extract coevo pairs
    if cuda != None:
        assert not os.system(f"\
            python {extract_coevo_pairs_script} \
            --input_msa {input_msa} \
            --output_pairs {coevo_file} \
            --cuda {cuda} \
        ")
    else:
        assert not os.system(f"\
            python {extract_coevo_pairs_script} \
            --input_msa {input_msa} \
            --output_pairs {coevo_file} \
        ")

    # extract esm2 encoding
    if cuda != None:
        assert not os.system(f"\
            python {extract_esm2_script} \
            --input_fasta {input_fasta} \
            --output_dir {esm2_dir} \
            --output_esm2 {esm2_file} \
            --cuda {cuda} \
        ")
    else:
        assert not os.system(f"\
            python {extract_esm2_script} \
            --input_fasta {input_fasta} \
            --output_dir {esm2_dir} \
            --output_esm2 {esm2_file} \
        ")

    # predict pairs
    if to_image:
        assert not os.system(f"\
            python {predict_script} \
            --input_encoding_file {esm2_file} \
            --input_pairs {coevo_file} \
            --output_result {output_pairs} \
            --to_image \
        ")
    else:
        assert not os.system(f"\
            python {predict_script} \
            --input_encoding_file {esm2_file} \
            --input_pairs {coevo_file} \
            --output_result {output_pairs} \
        ")
    
    # predict metal type
    if metal_type:
        assert not os.system(f"\
            python {predict_metal_type_script} \
                --input_pairs {output_pairs} \
                --output_pair {output_pairs} \
                --input_encoding_file {esm2_file} \
            ")

    # clean
    if not keep_inter_files:
        os.remove(coevo_file)
        os.remove(esm2_file)
        os.system(f"rm -rf {esm2_dir}")
        if msa_source == "colabfold":
            os.system(f"rm -rf {msa_dir}")
            os.remove(input_msa)
        if to_image:
            os.system(f"rm {osp.join(work_dir, '*.gv')}")


def main(argv):
    FLAGS = flags.FLAGS
    msa_source = FLAGS.msa_source
    input_fasta = FLAGS.input_fasta
    input_msa = FLAGS.input_msa
    output_pairs = FLAGS.output_pairs
    keep_inter_files = FLAGS.keep_inter_files
    to_image = FLAGS.to_image
    cuda = FLAGS.cuda
    metal_type = FLAGS.metal_type

    if msa_source == "user_defined":
        assert input_msa != None
    else:
        assert input_fasta != None
    run(
        input_fasta=input_fasta,
        input_msa=input_msa,
        msa_source=msa_source,
        output_pairs=output_pairs,
        keep_inter_files=keep_inter_files,
        to_image=to_image,
        cuda=cuda,
        metal_type=metal_type
    )


if __name__ == "__main__":
    define_arguments()
    app.run(main)
