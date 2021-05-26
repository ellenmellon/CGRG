set -e

DATA_DIR=../data/dstc



# prepare data
python generate_src_tgt_ctx_files_grounded.py
python compute_idf_dstc.py
python generate_constraints.py

python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_train.txt --output_fname $DATA_DIR/train.tsv --tokenizer pretrained/117M
python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_dev.txt --output_fname $DATA_DIR/dev.tsv --tokenizer pretrained/117M
python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_test.txt --output_fname $DATA_DIR/test.tsv --tokenizer pretrained/117M
python create_db.py --corpus $DATA_DIR/train.tsv --tokenizer pretrained/117M
