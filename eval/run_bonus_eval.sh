# Run the evaluation for generated responses of different bonus score
# Manually read the score and find the bonus score which gives the highest NIST score
# Then, use the selected bonus score to run the complete valid set 


cd /data2/ellen/cstr_grounded_conv/baselines/yizhe_gpt2/dstc

python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.0.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.01.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.02.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.03.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.04.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.05.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.06.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.07.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.08.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.09.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.10.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.11.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.12.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.13.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.14.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.15.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.2.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.25.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.3.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
python dstc.py /data2/ellen/experiments/dstc/attn_mask/cstr_doc_v2/small_dev.bonus_0.35.resp.txt /data2/ellen/experiments/dstc/small_ref.txt
