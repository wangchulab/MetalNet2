
source ~/conda.rc
conda activate metalnet

swiss_file_path=~/database/uniprot/swiss
anno_file_path=~/database/uniprot/swiss/metal_anno
species=('human' 'worm' 'yeast' 'zebrafish')


# extract ligands that contain metal element
for i in ${species[@]}
do
# swissprot file downloaded from https://www.uniprot.org/proteomes/ (protein count)
# e.g., for human, https://www.uniprot.org/proteomes/UP000005640, download all components as txt file type
python ./scripts/get_ligands.py --input ${swiss_file_path}/${i}.txt --output ${i}_ligands.csv
done

head -1 human_ligands.csv >> title # title
for i in ${species[@]}
do
sed '1d' ${i}_ligands.csv >> ligands.csv
done
sort -u ligands.csv -o ligands.csv
cat ligands.csv >> title && mv title ligands.csv
rm *_ligands.csv*
mv ligands.csv metal_ligands.csv

# then we manully check ligand (that does not have a inchi) if it's a metal-containing ligand
# and we manully add metal name to some specific metals

# annotate proteome at resi level (metal-binding)
for i in ${species[@]}
do
python ./scripts/annotate_proteome.py \
    --input_swiss ${swiss_file_path}/${i}.txt \
    --input_ligands metal_ligands.csv \
    --output ${anno_file_path}/${i}_metal_anno.csv
done