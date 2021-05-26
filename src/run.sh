set -e 

DATA_DIR=../data/
EXP_DIR=../saved_models/gpt2ia/


# train
python train_gpt2_distributed.py --config config_file/config-gr_cstr.json --gpu 0

# prediction
python run_gpt2.py --use_gpu --load_checkpoint <saved_model_checkpoint> --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file $DATA_DIR/test.tsv --output_file out --batch_size 64 --gpu 0 --beam --cstr_decode

