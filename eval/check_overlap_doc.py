'''
This script is used to calculate cumulative 
token count in generated responses vesus work rank.

run `python check_overlap_doc.py input_file (in decoded text) generated_response_file idf_file (third column as frequency)` 

It outputs a file, which will be read by the plot script under ../eval folder 

'''


import sys
import string
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))
PUNCS = set([i for i in string.punctuation])


def read_word2rank(fname):
    word2rank = {}
    with open(fname) as fin:
        for line in fin:
            if line.strip() == '':
                continue
            word, idf, freq = line.split('\t')
            rank = float(format(1.0 / int(freq), '.2f'))
            word2rank[word.strip()] = rank
    return word2rank

used_cstr = 0
n_cstr = 0
total_examples = 0
easy_examples = 0

doc_token_freq = 0
unique_doc_tokens = set()
unique_tokens = set()
rank2count = {}

# change this accordingly
token_in_doc = False

word2rank = read_word2rank(sys.argv[3])

input_setting = sys.argv[2].split('/')[-2]

with open(sys.argv[1]) as f_src, open(sys.argv[2]) as f_pred:
    src_lines = f_src.readlines()
    pred_lines = f_pred.readlines()
    n_lines = len(src_lines)
    for i in range(n_lines):
        src_line = src_lines[i]
        pred_line = pred_lines[i]
        src = src_line.strip(' \r\n')
        pred = pred_line.strip(' \r\n').split('\t')[-1]

        doc_and_cstr = src.split('<|endoftext|>')[-1].strip()
        if len(doc_and_cstr) == 0:
            continue
        src_tokens = ' '.join([t.strip() for t in src.split('<|endoftext|>')[:-1]]).split()
        cstr_tokens = set(doc_and_cstr.split('<s>')[-1].strip().split())
        cstr = [c.strip() for c in doc_and_cstr.split('<s>')[-1].strip().split('<c>') if len(c.strip()) > 0]
        doc_tokens = set(doc_and_cstr.split())
        pred_tokens = pred.strip().split()
        if len(cstr) > 0:
            total_examples += 1
        easy = False
        for t in cstr:
            if  t in pred.strip():
                easy = True
                used_cstr += 1
            n_cstr += 1
        if easy:
            easy_examples += 1

        '''
        else:
            if len(cstr) > 0:
                print(i, 'hard')
                print(src)
                print(pred)
        '''
        for t in set(pred_tokens):
            if t not in STOPWORDS and t not in PUNCS and t in doc_tokens and t not in cstr_tokens:
                doc_token_freq += 1
                unique_doc_tokens.add(t)
            if t not in STOPWORDS and t not in PUNCS and t not in cstr_tokens:
                unique_tokens.add(t)

    if token_in_doc:
        for t in unique_doc_tokens:
            if t in word2rank:
                rank = word2rank[t]
                if rank in rank2count:
                    rank2count[rank] += 1
                else:
                    rank2count[rank] = 1
    else:
        for t in unique_tokens:
            if t in word2rank:
                rank = word2rank[t]
                if rank in rank2count:
                    rank2count[rank] += 1
                else:
                    rank2count[rank] = 1

    print(used_cstr*1.0/n_cstr)
rank2cumcount = {}
total = 0
for key in sorted(rank2count.keys(), reverse=True):
    total += rank2count[key]
    rank2cumcount[key] = total

with open('../eval/vocab_coverage_vs_rank_{}.txt'.format(input_setting), 'w') as fout:
    for rank in sorted(rank2cumcount):
        fout.write('{},{}\n'.format(rank, rank2cumcount[rank]))
    
