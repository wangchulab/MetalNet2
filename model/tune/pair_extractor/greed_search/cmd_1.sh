source ~/conda.rc
conda activate metalnet

script_path=`realpath "../../../scripts/"`
msa_file_path="~/database/metalnet/msa/"
params_file=`realpath "../../default.yaml"`
anno_file=`realpath "../../../../dataset/transform/train_metalnet.tsv"`

encode_method=esm2
encode_strategy=avg
esm2_encoding_file="~/database/metalnet/encoding/esm2_files.tsv"

# msa_filter_type searching

coevo_threshold=0.1
msa_filter_types=(hhfilter) # hamming was tested before
num_seq=64
msa_type=uni

result_path="./result/msa_filter/"
mkdir -p $result_path
cd $result_path

for m in ${msa_filter_types[@]}
do
record_file=$m.tsv
python $script_path/extract_coevo_pairs.py \
    --input_msa $msa_file_path/train_msa_files.$msa_type.csv \
    --output_pairs $record_file \
    --coevo_threshold $coevo_threshold \
    --msa_filter_type $m \
    --num_seq $num_seq \
    --cuda 0
done

for m in ${msa_filter_types[@]}
do
record_file=$m.tsv
result_file=${m}_result.tsv
python $script_path/tune_model.py \
    --input_params $params_file \
    --input_records $record_file \
    --input_anno $anno_file \
    --input_encoding_file $esm2_encoding_file \
    --output_result $result_file \
    --encode_method $encode_method \
    --encode_strategy $encode_strategy
done