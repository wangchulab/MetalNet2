source ~/conda.rc
conda activate metalnet

params_file="../default.yaml"
encoding_file="~/database/metalnet/encoding/esm2_files.tsv"
pred_pair_file="../tmp/train_pred_pairs.tsv"
anno_file="../../../dataset/transform/train_metalnet.tsv"
script_file="../../scripts/tune_model.py"
strategy=(avg cat max)
encode_method=esm2

mkdir -p ./result/
for s in ${strategy[@]}
do
result_file=./result/${encode_method}_${strategy}_result.tsv
python $script_file \
    --input_params $params_file \
    --input_records $pred_pair_file \
    --input_anno $anno_file \
    --input_encoding_file $encoding_file \
    --encode_method $encode_method \
    --encode_strategy $s \
    --output_result $result_file
done