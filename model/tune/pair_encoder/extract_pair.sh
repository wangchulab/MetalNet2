source ~/conda.rc
conda activate metalnet

SCRIPTS_PATH="../../scripts/"

# default params
msa_type=uni # uniref
msa_filter_type=hamming
num_seq=64
coevo_threshold=0.1

msa_file="~/database/metalnet/msa/train_msa_files.$msa_type.csv"

mkdir tmp
python $SCRIPTS_PATH/extract_coevo_pairs.py \
    --input_msa $msa_file \
    --msa_filter_type $msa_filter_type \
    --num_seq $num_seq \
    --coevo_threshold $coevo_threshold \
    --cuda 0 \
    --output_pairs "./tmp/coevo_pairs.tsv"