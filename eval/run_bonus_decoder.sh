# Run the best model on smaller dataset by tuning different bonus score
# save predicted file to the same result folder



python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.2 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.2

python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.25 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.25

python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.3 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.3

python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.35 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.35

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.05 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.05

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.06 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.06

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.07 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.07

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.08 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.08

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.09 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.09

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.10 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.10

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.11 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.11

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.12 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.12

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.13 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.13

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.14 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.14

#python run_gpt2.py --use_gpu --load_checkpoint /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/GPT2.1e-05.32.4gpu.2019-08-22061032/GP2-pretrain-step-6501.pkl --model_name_or_path pretrained/117M/ --max_seq_length 512 --generation_length 90 --test_file /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.tsv --output_file bonus_0.15 --batch_size 64 --gpu 1 --beam --beam_width 1 --cstr_decode --bonus 0.15
