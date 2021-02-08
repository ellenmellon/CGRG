DATA_DIR=../data/
EXP_DIR=../saved_models/gpt2ia/

# prepare data
# it is using toy data examples only for now.
python ../prepare_data/generate_input.py --cstr_fname $DATA_DIR/toy_raw_train.txt --output_fname $DATA_DIR/train.tsv --tokenizer pretrained/117M
python ../prepare_data/generate_input.py --cstr_fname $DATA_DIR/toy_raw_dev.txt --output_fname $DATA_DIR/dev.tsv --tokenizer pretrained/117M
python ../prepare_data/generate_input.py --cstr_fname $DATA_DIR/toy_raw_test.txt --output_fname $DATA_DIR/test.tsv --tokenizer pretrained/117M
python create_db.py --corpus $DATA_DIR/train.tsv --tokenizer pretrained/117M


# train
python train_gpt2_distributed.py --config config_file/config_train_gpt2ia/config-gr_cstr.json --gpu 0

# prediction
python run_gpt2.py --use_gpu --load_checkpoint $EXP_DIR/model.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file $DATA_DIR/test.tsv --output_file out --barch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode

