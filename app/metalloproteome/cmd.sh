source ~/conda.rc
conda activate metalnet

script_path="../../model/scripts/"

coevo_threshold=0.1
msa_filter_type=hamming
num_seq=64
msa_type=uni

species=(human worm zebrafish yeast)

# coevo pairs
for s in ${species[@]}
do
coevo_pairs_file=${s}_coevo_pairs.tsv
msa_file=~/database/msa/${s}_msa_files.tsv

python $script_path/extract_coevo_pairs.py \
    --input_msa $msa_file \
    --output_pairs $coevo_pairs_file \
    --coevo_threshold $coevo_threshold \
    --msa_filter_type $msa_filter_type \
    --num_seq $num_seq \
    --cuda 0 >> cmd.log 2>&1
done

# pred result
for s in ${species[@]}
do
input_encoding_file="~/database/encoding/esm2/${s}_esm_files.tsv"
coevo_pairs_file=${s}_coevo_pairs.tsv
pred_pairs_file=${s}_pred_pairs.tsv

python $script_path/predict_pairs.py \
    --input_pairs $coevo_pairs_file \
    --input_encoding_file $input_encoding_file \
    --output_result $pred_pairs_file
done



# clean and extract positive

database_path=~/database/metalnet/prediction/
for s in ${species[@]}
do
pred_pairs_file=${s}_pred_pairs.tsv
coevo_pairs_file=${s}_coevo_pairs.tsv
pred_true_file=${s}_pred_mbps.tsv
awk 'NR == 1 || (NR > 1 && $NF == 1)' $pred_pairs_file > $pred_true_file
mv $pred_pairs_file $coevo_pairs_file -t $database_path
done