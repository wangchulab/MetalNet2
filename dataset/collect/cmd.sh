source ~/conda.rc
conda activate metalnet

python ./scripts/get_metal_binding_proteins.py \
    --input ~/database/pdb/rcsb_20230430/all_protein_files.csv \
    --output_records mbp_files.tsv \
    --output_fasta mbps.fasta \
    --resolution_threshold 3.0 >> cmd.log 2>&1
python ./scripts/get_metal_binding_residues.py \
    --input mbp_files.tsv \
    --include_metal_compound \
    --include_common_metal_only \
    --include_water \
    --include_main_chain \
    --output metal_sites.tsv >> cmd.log 2>&1
python ./scripts/filter_by_seq_identity.py \
    --input_fasta mbps.fasta \
    --input_pdb mbp_files.tsv \
    --input_records metal_sites.tsv \
    --output_records nr_metal_sites.tsv \
    --output_fasta nr_metal_sites.fasta \
    --output_pdb nr_mbp_files.tsv >> cmd.log 2>&1

python ./scripts/transfer_annotations.py \
    --input_records nr_metal_sites.tsv \
    --input_pdb nr_mbp_files.tsv \
    --output_records metal_chains.tsv >> cmd.log 2>&1
python ./scripts/assign_domains_by_struct.py \
    --input_records metal_chains.tsv \
    --input_pdb nr_mbp_files.tsv \
    --input_fasta nr_metal_sites.fasta \
    --output_records metal_chains.tsv \
    --output_fasta metal_chains.fasta >> cmd.log 2>&1
python ./scripts/replace_non_common_aa.py \
    --input_fasta metal_chains.fasta \
    --output_fasta metal_chains.fasta >> cmd.log 2>&1
python ./scripts/split_train_test.py \
    --input_records metal_chains.tsv \
    --output_records metal_chains.tsv \
    --output_train_records metal_chains_train.tsv \
    --output_test_records metal_chains_test.tsv >> cmd.log 2>&1

# copy to database
cp metal_chains.fasta ~/database/metalnet/uniprot/fasta/metal_chains.fasta

# move to tmp
mkdir tmp
mv mbp_files.tsv mbps.fasta metal_sites.tsv nr_mbp_files.tsv nr_metal_sites.fasta -t tmp/