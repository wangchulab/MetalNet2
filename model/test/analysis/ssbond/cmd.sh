source ~/conda.rc
conda activate metalnet

PROJECT_DIR="../../../../"
python ./scripts/get_interacted_pairs.py \
    --input_pdb $PROJECT_DIR/dataset/collect/tmp/nr_mbp_files.tsv \
    --input_records $PROJECT_DIR/dataset/transform/test_metalnet.tsv \
    --output_records test_disulfide_pairs.tsv > cmd.log 2>&1