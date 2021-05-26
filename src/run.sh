set -e 

DATA_DIR=../data/dstc
EXP_DIR=../saved_models/gpt2ia/



# train
python create_db.py --corpus $DATA_DIR/train.tsv --tokenizer pretrained/117M
python train_gpt2_distributed.py --config config_file/config-gr_cstr.json --gpu 0

# prediction
python run_gpt2.py --use_gpu --gpu 0 --beam --load_checkpoint ../saved_models/model.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file $DATA_DIR/test.tsv --output_file out --batch_size 256 --output_ref --cstr_decode

# prediction for toy file
python run_gpt2.py --use_gpu --gpu 0 --beam --load_checkpoint ../saved_models/model.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file $DATA_DIR/test_toy.tsv --output_file out_toy --batch_size 256 --output_ref --cstr_decode
