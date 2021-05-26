set -e

DATA_DIR=../data/dstc


# prepare data
python generate_src_tgt_ctx_files_grounded.py
python compute_idf_dstc.py
python generate_constraints.py
python prune.py
