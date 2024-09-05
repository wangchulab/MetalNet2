source ~/conda.rc
conda activate metalnet

script_path="../scripts/"
msa_file_path="~/database/metalnet/msa"
esm2_file="~/database/metalnet/encoding/esm2_files.tsv"
anno_file="../../dataset/transform/train_metalnet.tsv"
train_params_file="./params.yaml"

# best params through tuning
encode_method=esm2
encode_strategy=avg

pred_pairs_file="./tmp/train_pred_pairs.tsv"
model_path=./MetalNet_metal_type_AutogluonModels

# cp "../tune/tmp/" ./ -r

python $script_path/train_model.py \
    --input_params $train_params_file \
    --input_records $pred_pairs_file \
    --input_anno $anno_file \
    --input_encoding_file $esm2_file \
    --encode_method $encode_method \
    --encode_strategy $encode_strategy \
    --output_model $model_path > cmd.log 2>&1