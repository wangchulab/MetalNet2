import os.path as osp


params_file = osp.abspath("../default.yaml")
anno_file = osp.abspath("../../../dataset/transform/train_metalnet.tsv")
coevo_pair_file = osp.abspath("./tmp/coevo_pairs.tsv")
scripts_path = osp.abspath("../../scripts/")


def build_params(
    prefix: str,
    encode_method: str,
    encode_strategy: str,
    encoding_file: str,
):
    return {
        "prefix": prefix,
        "workdir": osp.abspath(f"./result/{prefix}"),
        "encode_method": encode_method,
        "encode_strategy": encode_strategy,
        "encoding_file": encoding_file,
        "tuning_result_file": f"{prefix}_result.tsv",
    }


# msa type
msa_type = "uni"

# af2 file to choose
af2_model_type = ['rank', 'model']
af2_model_num = range(1, 6)

# nb for pair
nb_strategy = "centroid"

# single resi encoding to pair encoding
single_to_pair = ["avg", "max", "cat"]

pair_encoding_methods = ["freq_mtx", "af2_pair_repr", "nb"]
resi_encoding_methods = ["af2_resi_repr", "esm2", "nb"]

job_params = []
for m in pair_encoding_methods:
    if m == "freq_mtx":
        prefix = f"{m}_{msa_type}"
        job_params.append(build_params(
            prefix=prefix,
            encode_method=m,
            encode_strategy=None,
            encoding_file=f"~/database/metalnet/msa/train_msa_files.{msa_type}.csv",
        ))
    elif m == "af2_pair_repr":
        for s in af2_model_type:
            for n in af2_model_num:
                prefix = f"{m}_{s}_{n}"
                job_params.append(build_params(
                    prefix=prefix,
                    encode_method=m,
                    encode_strategy=None,
                    encoding_file=f"~/database/metalnet/encoding/af2_{s}_{n}_files_pair_repr.tsv",
                ))
    elif m == "nb":
        prefix = f"{m}_{nb_strategy}"
        job_params.append(build_params(
            prefix=prefix,
            encode_method=m,
            encode_strategy=nb_strategy,
            encoding_file="~/database/metalnet/pdb/metalnet_pdb_files.tsv",
        ))

for m in resi_encoding_methods:
    if m == "af2_resi_repr":
        for t in af2_model_type:
            for n in af2_model_num:
                for s in single_to_pair:
                    prefix = f"{m}_{t}_{n}_{s}"
                    job_params.append(build_params(
                        prefix=prefix,
                        encode_method=m,
                        encode_strategy=s,
                        encoding_file=f"~/database/metalnet/encoding/af2_{t}_{n}_files_resi_repr.tsv",
                    ))
    elif m == "esm2":
        for s in single_to_pair:
            prefix = f"{m}_{s}"
            job_params.append(build_params(
                prefix=prefix,
                encode_method=m,
                encode_strategy=s,
                encoding_file="~/database/metalnet/encoding/esm2_files.tsv",
            ))
    elif m == "esm1b":
        for s in single_to_pair:
            prefix = f"{m}_{s}"
            job_params.append(build_params(
                prefix=prefix,
                encode_method=m,
                encode_strategy=s,
                encoding_file="~/database/metalnet/encoding/esm1b_files.tsv",
            ))
    elif m == "nb":
        for s in single_to_pair:
            prefix = f"{m}_{s}"
            job_params.append(build_params(
                prefix=prefix,
                encode_method=m,
                encode_strategy=s,
                encoding_file="~/database/metalnet/pdb/metalnet_pdb_files.tsv",
            ))

tune_cmd_lines = []
tune_cmd_lines.append(
    '''
source ~/conda.rc
conda activate metalnet
'''
)

for job in job_params:

    tune_cmd_lines.append(
        f'''
mkdir -p {job['workdir']}
cd {job['workdir']}
python {scripts_path}/tune_model.py \
--input_params {params_file} \
--input_records {coevo_pair_file} \
--input_anno {anno_file} \
--input_encoding_file {job['encoding_file']} \
--encode_method {job['encode_method']} \
--encode_strategy {job['encode_strategy']} \
--output_result {job['tuning_result_file']}
'''
    )

with open("tune.sh", "w") as f:
    f.writelines(tune_cmd_lines)
