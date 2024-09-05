source ~/conda.rc
conda activate metalnet

scripts_file="../../../model/scripts/run_prediction_workflow.py"
input_path="./input/"
result_path="./output"

mkdir -p $result_path
python $scripts_file \
    --input_fasta "$input_path/example.fasta" \
    --msa_source colabfold \
    --output_pairs $result_path/pred_pairs.tsv \
    --to_image \
    --cuda 1