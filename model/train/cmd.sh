source ~/conda.rc
conda activate metalnet

script_path="../scripts/"
msa_file_path="~/database/metalnet/msa"
esm2_file="~/database/metalnet/encoding/esm2_files.tsv"
anno_file="../../dataset/transform/train_metalnet.tsv"
train_params_file="./params.yaml"

# best params through tuning
coevo_threshold=0.1
msa_filter_type=hamming
num_seq=64
msa_type=uni
encode_method=esm2
encode_strategy=avg

coevo_pairs_file=coevo_pairs.tsv
model_path=./MetalNet_AutogluonModels

python $script_path/extract_coevo_pairs.py \
    --input_msa $msa_file_path/train_msa_files.$msa_type.csv \
    --output_pairs $coevo_pairs_file \
    --coevo_threshold $coevo_threshold \
    --msa_filter_type $msa_filter_type \
    --num_seq $num_seq \
    --cuda 0

python $script_path/train_model.py \
    --input_params $train_params_file \
    --input_records $coevo_pairs_file \
    --input_anno $anno_file \
    --input_encoding_file $esm2_file \
    --encode_method $encode_method \
    --encode_strategy $encode_strategy \
    --output_model $model_path > cmd.log 2>&1

mkdir -p tmp/
mv $coevo_pairs_file -t tmp/