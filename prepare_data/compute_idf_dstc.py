# each token will be lowered to compute idf

from collections import defaultdict
from nltk.corpus import stopwords
import string
import math

STOPWORDS = set(stopwords.words('english'))
PUNCS = set([i for i in string.punctuation])
PROJECT_FOLDER = '../'
DATASET = 'dstc'



train_file = '{}/data/{}/src_tgt_ctx/train_target.txt'.format(PROJECT_FOLDER, DATASET)

nDocs = 0
doc2WordSet = defaultdict(set)
word2freq = defaultdict(int)
with open(train_file) as fin:
    for line in fin:
        words = [t.lower() for t in line.strip(' \r\n').split('\t')[-1].split()]
        doc2WordSet[nDocs] = set(words)
        for t in words:
            word2freq[t] += 1
        nDocs += 1

word2NumDocs = defaultdict(int)
for doc in doc2WordSet:
    for word in doc2WordSet[doc]:
        word2NumDocs[word] += 1

word2IDF = {}
for key in word2NumDocs:
    idf = math.log(nDocs*1.0/word2NumDocs[key])
    if idf > 0:
        word2IDF[key] = idf

word2IDF = sorted(word2IDF.items(), key=lambda kv: kv[1])#, reverse=True)

idf_file = '{}/data/dstc/idf_files/idf_{}.txt'.format(PROJECT_FOLDER, DATASET)
with open(idf_file, 'w') as fout:
    for t in word2IDF:
        if t[0] in STOPWORDS or t[0] in PUNCS:
            continue
        fout.write('{}\t{}\t{}\n'.format(t[0], t[1], word2freq[t[0]]))
