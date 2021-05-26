"""
remove no cstr or too long examples 

"""

import sys
import spacy
import re
import time
import math
import random
from multiprocessing import Process, Lock

PROJECT_FOLDER = '../'
DATASET = 'dstc'
PREFICES = ['train', 'dev', 'test']
N_THREADS = 20

class SpacyTokenizer():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    def tokenize(self, text):
        clean_text = ' '.join(text.strip().split())
        sents = [sent for sent in self.nlp(clean_text).sents]
        sents = [' '.join([token.text for token in sent]) for sent in sents]

        sents = list(filter(lambda x: len(x) > 0, sents))
        return sents

tokenizer = SpacyTokenizer()



def getCleanAndNoisySents(tokenizer, context, curr_constraint, context_name):
    noisy = []
    clean = []
    for sent in tokenizer.tokenize(context):
        if curr_constraint.lower() not in sent.lower():
            noisy += ['{} {}'.format(context_name, sent)]
            continue

        clean += [('{} {}'.format(context_name, sent), curr_constraint)]
    return clean, noisy


def writePrunedExamplesToFile(tokenizer, examples, start_idx, end_idx, fout, lock, proc_id):
    random.seed(1234)
    ct_neg = 0
    ct_pos = 0
    start_time = time.time()
    for i in range(start_idx, min(len(examples), end_idx)):
        if i%1000 == 0:
            print(proc_id, i-start_idx, time.time()-start_time, flush=True)
        e = examples[i]
        lines = e.split('\n')
        if len(lines) <= 3:
            continue


        idx = 3
        to_write = []
        all_noisy_tmp = []
        all_constraints = set()
        all_clean_sents = set()
        while idx < len(lines):
            context = ' '.join(lines[idx].strip(' \r\n').split()[1:])
            context_name = lines[idx].strip(' \r\n').split()[0]
            constraint = lines[idx+1].strip(' \r\n')
            #sents = ['{} {}'.format(context_name, sent) for sent in tokenizer.tokenize(context) if constraint in sent]
            clean_sents, noisy_sents = getCleanAndNoisySents(tokenizer, context, constraint, context_name)

            all_noisy_tmp += noisy_sents
            all_constraints.add(constraint)

            to_write += clean_sents
            for s in clean_sents:
                all_clean_sents.add(s[0])

            idx += 2

    
        if len(to_write) == 0:
            continue

        all_noisy_tmp = set(all_noisy_tmp)
        all_noisy = all_noisy_tmp.difference(all_clean_sents)
    

        lock.acquire()    
        fout.write(str('\n'.join(lines[:3])+'\n').encode("utf-8"))
        for pair in to_write:
            s, constraint = pair
            fout.write(str(s+'\n').encode("utf-8"))
            fout.write(str(constraint+'\n').encode("utf-8"))
        for s in all_noisy:
            fout.write(str(s+'\n').encode("utf-8"))
            fout.write(str('n/a\n').encode("utf-8"))

        fout.write(str('\n').encode("utf-8"))
        lock.release()
        ct_pos += 1
    print(time.time()-start_time)
    print('percentage of examples with no constraint : {}'.format(1.0*ct_neg/(ct_neg+ct_pos)))  


for prefix in PREFICES:

    with open('{}/data/dstc/conv_with_cstr/constraints_{}.txt'.format(PROJECT_FOLDER, prefix)) as fin:
        examples = fin.read().split('\n\n')

    # creating training data for QA andf write to output file
    lock = Lock()
    processes = []
    instancesPerProc = int(math.ceil(len(examples)*1.0/N_THREADS))
    fout = open('{}/data/dstc/conv_with_cstr/constraints_{}_clean.txt'.format(PROJECT_FOLDER, prefix), 'wb', buffering=0)
    
    for proc_id in range(N_THREADS):
        start_idx = proc_id * instancesPerProc
        end_idx = (proc_id+1) * instancesPerProc
        args = (tokenizer, examples, start_idx, end_idx, fout, lock, proc_id)
        p = Process(target=writePrunedExamplesToFile, args=args)
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    fout.close()


    
