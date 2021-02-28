import pandas as pd
import glob
import os
import argparse

def read_eval_file(eval_path):
    lines = open(eval_path).readlines()[4:]
    exp_name = os.path.splitext(os.path.basename(eval_path))[0]
    result_dic = {}
    exec("\n".join(lines), None, result_dic)
    row = [exp_name] + result_dic["NIST"] + result_dic["BLEU"] + [result_dic["METEOR"]] + \
          result_dic["entropy"] + result_dic["diversity"] + [result_dic["avg_len"]]
    return row


def gen_col_name(prefix, n = 4):
    return [prefix + str(i) for i in range(1, 1+n)]



parser = argparse.ArgumentParser(description="""
Give a directory, enumerate all *.eval.txt under it, parse the output.
Generate a dataframe, copy it to clipboard
""")
parser.add_argument("--eval_dir", type=str, default="/home/yizhe/data/GPT-2/dstc/eval")
args = parser.parse_args()

rows = []
for fn in glob.glob(os.path.join(args.eval_dir, "*.eval.txt")):
    row = read_eval_file(fn)
    rows.append(row)

df = pd.DataFrame(rows, columns=["Experiment"] + gen_col_name("NIST") +
                           gen_col_name("BLEU") +
                           ["METEOR"] +
                           gen_col_name("entropy") +
                           gen_col_name("diversity", 2) +
                           ["avg_len"]
             )
df.to_csv(os.path.join(args.eval_dir, "summary.csv"))

# df.to_clipboard(index=False)