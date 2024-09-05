# coevo pair

bash extract_pair.sh


# tune

source ~/conda.rc
conda activate metalnet

python gen_bash_scripts.py
bash tune.sh


# clean

rm tune.sh
find ./result/ ! -name *result.tsv | xargs rm -rf