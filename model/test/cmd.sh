source ~/conda.rc
conda activate metalnet

script_path="../scripts/"
msa_file_path="~/database/metalnet/msa"
esm2_file="~/database/metalnet/encoding/esm2_files.tsv"

# best params through tuning
coevo_threshold=0.1
msa_filter_type=hamming
num_seq=64
msa_type=uni
encode_method=esm2
encode_strategy=avg
prob_threshold=0.8
node_num_threshold=4

coevo_pairs_file=coevo_pairs.tsv
model_path="../train/MetalNet_AutogluonModels"
pred_pairs_file=pred_pairs.tsv

python $script_path/extract_coevo_pairs.py \
    --input_msa $msa_file_path/train_msa_files.$msa_type.csv \
    --output_pairs $coevo_pairs_file \
    --coevo_threshold $coevo_threshold \
    --msa_filter_type $msa_filter_type \
    --num_seq $num_seq \
    --cuda 0

python $script_path/predict_pairs.py \
    --input_encoding_file $esm2_file \
    --input_pairs $coevo_pairs_file \
    --output_result $pred_pairs_file \
    --model_path $model_path \
    --encode_method $encode_method \
    --encode_strategy $encode_strategy \
    --prob_threshold $prob_threshold \
    --node_num_threshold 4

mkdir -p tmp/
mv $coevo_pairs_file -t tmp/