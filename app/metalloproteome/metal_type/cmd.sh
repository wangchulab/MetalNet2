source ~/conda.rc
conda activate metalnet

script_path="../../../extra/scripts/predict_metal_type.py"

species=(human worm zebrafish yeast)

# pred result
for s in ${species[@]}
do
input_encoding_file="~/database/encoding/esm2/${s}_esm_files.tsv"
pred_pairs_file="../${s}_pred_mbps.tsv"
output_file=${s}_pred_mbps_with_type.tsv

python $script_path \
    --input_pairs $pred_pairs_file \
    --input_encoding_file $input_encoding_file \
    --output_resi $output_file
done