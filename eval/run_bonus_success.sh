# Run the success rate for different bonus score on small valid set.
# So then, afterwards, can plot success rate versus NIST/BLEU score to show that 
# there is a trade-off between strict constraint inclusion and performance 

python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.0.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.05.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.1.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.15.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.2.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.25.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.3.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.35.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.4.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.45.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.5.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.55.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.6.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.13.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.14.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.15.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.16.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.17.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.18.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.19.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.2.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
#python check_overlap_doc.py /data2/ellen/experiments/dstc/small_input.txt /data2/ellen/experiments/dstc/attn_mask/no_cstr_v3_reset/small_dev.bonus_0.25.resp.txt /data2/ellen/experiments/dstc/idf_dstc.txt
