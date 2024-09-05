
# see analyze_pred_evidence.ipynb

# 1.1
source ~/conda.rc
conda activate metalnet

script_path=`realpath "../../../dataset/collect/scripts/"`
cd tmp/
 
# python $script_path/get_metal_binding_proteins.py \
#     --input ~/database/pdb/rcsb_20230430/all_protein_files.csv \
#     --output_records exp_mbp_files.tsv \
#     --output_fasta exp_mbps.fasta >> cmd.log 2>&1 # do not set resolution threshold
# 
# python $script_path/get_metal_binding_residues.py \
#     --input exp_mbp_files.tsv \
#     --output exp_metal_sites.tsv \
#     --include_metal_compound \
#     --include_common_metal_only \
#     --include_water \
#     --include_main_chain \
#     --interacted_distance 4 \
#     --interacted_num_resi 2 \
#     --use_sloppy_mode >> cmd.log 2>&1


# 1.3
BLAST_EXE_PATH=~/install/ncbi-blast-2.14.0+/bin
# ${BLAST_EXE_PATH}/makeblastdb -in exp_metal_polymers.fasta -parse_seqids -hash_index -dbtype prot -out ./blastdb/exp_metal_polymers.db

# 1.4
# ${BLAST_EXE_PATH}/blastp -query pred_mbps.fasta -out pred_mbps_exp.blast -db ./blastdb/exp_metal_polymers.db -outfmt 6 -evalue 1e-5 -num_threads 24

# 2.2
${BLAST_EXE_PATH}/makeblastdb -in anno_metal_uniprots.fasta -parse_seqids -hash_index -dbtype prot -out ./blastdb/anno_metal_uniprots.db

# 2.3
${BLAST_EXE_PATH}/blastp -query pred_mbps.fasta -out pred_mbps_anno.blast -db ./blastdb/anno_metal_uniprots.db -outfmt 6 -evalue 1e-5 -num_threads 24
# 
# find . ! -name 'cmd.sh' | xargs rm -rf