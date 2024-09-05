## Install

The project runs under Ubuntu 18.04.6, Linux-5.3.0-47-generic-x86_64-with-glibc2.27.

### prepare python env
```bash
conda env create -n metalnet -f env.yaml
```

If it doesn't work, try:
```bash
conda create -n metalnet python=3.9.16
conda activate metalnet
conda install python-graphviz==0.20.1
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install hhsuite==3.3.0 -c conda-forge -c bioconda
pip install autogluon.tabular[all]==0.8.0
pip install biopython==1.81
pip install fair-esm==2.0.0
pip install absl-py
pip install more-itertools
```

### download model weights
1. Download [model_1](https://drive.google.com/file/d/1Qihh1iqIosVKZO0-OH800qj0eXAvCfMu/view?usp=sharing) for metal-binding prediction, [model_2](https://drive.google.com/file/d/1GSOpS0AFeraaRUTWoFQS-hiVNh1p8Azr/view?usp=sharing) for metal-type prediction.
2. Place model_1 at ./model/train/ and model_2 at ./extra/train/, then unzip them and remove zip files.

## Usage

See ./app/example/.


## Optional

Optional python packages:
```bash
conda install mmseqs2 -c conda-forge -c bioconda
pip install matplotlib_venn==0.11.9
pip install bs4==0.0.1
pip install lxml==4.9.2
```

Optional executables:
```bash
# blast related
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.14.0+-x64-linux.tar.gz
```

Patches:
```bash
PACKAGE_PATH="/path/to/site-packages"

# add arial related font to matplotlib
cp "./asset/*.ttf" -t $PACKAGE_PATH/matplotlib/mpl-data/fonts/ttf/
rm -r "~/.cache/matplotlib/"
```

Add `PROJECT_DIR = "/path/to/metalnet"` in `~/.ipython/profile_default/ipython_config.py` as a global variable for use.

We find a bug in [MMCIFParser](https://github.com/biopython/biopython/pull/4399), and it will result in a loss of about 100 chains in dataset.
But in this project we use the old version of MMCIFParser, still.