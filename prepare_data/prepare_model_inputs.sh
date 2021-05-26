set -e 

DATA_DIR=../data/dstc


python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_train_clean.txt --output_fname $DATA_DIR/train.tsv --tokenizer ../src/pretrained/117M
python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_dev_clean.txt --output_fname $DATA_DIR/dev.tsv --tokenizer ../src/pretrained/117M
python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_test_clean.txt --output_fname $DATA_DIR/test.tsv --tokenizer ../src/pretrained/117M

# toy test file
python generate_input.py --cstr_fname $DATA_DIR/conv_with_cstr/constraints_test_toy.txt --output_fname $DATA_DIR/test_toy.tsv --tokenizer ../src/pretrained/117M