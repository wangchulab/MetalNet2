source ~/conda.rc
conda activate metalnet

input_pairs=./tmp/pred_pairs.tsv
output=pred_metal_type.tsv
input_encoding_file="~/database/metalnet/encoding/esm2_files.tsv"

mkdir -p tmp/
cp "../../model/test/pred_pairs.tsv" $input_pairs

python "../scripts/predict_metal_type.py" \
    --input_pairs $input_pairs \
    --output_resi $output \
    --input_encoding_file $input_encoding_file