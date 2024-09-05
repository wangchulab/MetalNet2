source ~/conda.rc
conda activate metalnet

result_file=./tmp/train_pred_pairs.tsv
mkdir -p tmp
python "../../model/scripts/predict_pairs.py" \
    --input_encoding_file "~/database/metalnet/encoding/esm2_files.tsv" \
    --input_pairs "../../model/train/tmp/coevo_pairs.tsv" \
    --output_result $result_file

awk 'NR == 1 || $NF == 1 {print $0}' $result_file > tmp.tsv
mv tmp.tsv $result_file