source ~/conda.rc
conda activate metalnet

python ./scripts/clean_dataset_for_metalnet.py \
    --input_records ../collect/metal_chains.tsv \
    --input_fasta ../collect/metal_chains.fasta \
    --output_train_records train_metalnet.tsv \
    --output_test_records test_metalnet.tsv \
    --output_fasta metal_chains_metalnet.fasta >> cmd.log 2>&1