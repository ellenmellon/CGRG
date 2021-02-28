####################
# 
##################

import sys
import string
from nltk.corpus import stopwords
from collections import defaultdict
import random
import re
import time
import math
from multiprocessing import Process, Lock
import operator

STOPWORDS = set(stopwords.words('english'))
PUNCS = set([i for i in string.punctuation])
MAX_N_MATCHES=20
MAX_PHRASE_LENGTH=5
N_THREADS=20
dataset_2_min_mean_idf = {'dstc':8.5}

PREFICES = ['train', 'dev', 'test']
DATASET = 'dstc'

PROJECT_FOLDER = '/data2/ellen/cstr_grounded_conv'


def loadWord2IDF(fname):
    word2IDF = defaultdict(float)
    with open(fname) as fin:
        for line in fin:
            key_idf =line.strip('\r\n').split('\t')
            word = key_idf[0]
            idf = float(key_idf[1])
            word2IDF[word] = idf
    return word2IDF

def getMeanIDFScore(phrase, word2IDF):
    sum_IDF = 0.0
    tokens = phrase.split()
    for token in tokens:
        sum_IDF += word2IDF[token]
    return sum_IDF / len(tokens)

def isValidMatchedPhrase(ctx_phrase, mean_idf, min_mean_idf=0.000):
    ctx_phrase_tokens = ctx_phrase.split()
    puncs_and_stopwords = PUNCS.union(STOPWORDS)
    
    first_token = ctx_phrase_tokens[0]
    last_token = ctx_phrase_tokens[-1]
    if last_token in puncs_and_stopwords or first_token in puncs_and_stopwords:
        return False
    elif len(ctx_phrase_tokens) > 1 or mean_idf > min_mean_idf:
        return True
    return False

def hasOverlap(start1, end1, start2, end2):
    return (min(end1, end2) - max(start1, start2)) > 0

def getBestMatchedPhrase(query, context_sents, response, word2IDF, dataset=None, max_phrase_length=5):
    response = response.split()
    phrasesInfo = []
    constraints = []
    min_mean_idf = dataset_2_min_mean_idf[dataset]

    # each sentence needs to be in the format of "<category> sent"
    for context in context_sents:
        context = context.split()

        for phrase_length in range(max_phrase_length, 0, -1):

            for i in range(len(response)-phrase_length+1):
                response_phrase = ' '.join(response[i:i+phrase_length])
                if response_phrase.lower() in query.lower():
                    continue

                for j in range(len(context)-phrase_length+1):
                    ctx_phrase = ' '.join(context[j:j+phrase_length])
                    if response_phrase.lower() != ctx_phrase.lower():
                        continue

                    mean_idf = getMeanIDFScore(ctx_phrase.lower(), word2IDF)

                    if isValidMatchedPhrase(ctx_phrase.lower(), mean_idf, min_mean_idf=min_mean_idf):
                        phraseInfo = (phrase_length, mean_idf, j, j+phrase_length, response_phrase, ' '.join(context))
                        phrasesInfo.append(phraseInfo)
                        constraints.append((i, i+phrase_length))

    phrasesInfo.sort(key=operator.itemgetter(0, 1), reverse=True)
    if len(phrasesInfo) == 0:
        return None, None

    result = [phrasesInfo[0]]
    c_result = [constraints[0]]

    for p1 in phrasesInfo[1:]:
        toAdd = True
        for p2 in result: 
            if p1[-1] == p2[-1] and hasOverlap(p1[2], p1[3], p2[2], p2[3]):
                toAdd = False
                break
        if toAdd:
            result.append(p1)

    for p1 in constraints[1:]:
        toAdd = True
        for p2 in c_result:
            if hasOverlap(p1[0], p1[1], p2[0], p2[1]):
                toAdd = False
                break
        if toAdd:
            c_result.append(p1)

    return result[:MAX_N_MATCHES], c_result

# function for multiprocessing
def writeMatchedPhrasesToFile(
        dataset,
        queries, 
        responses, 
        contexts,  
        word2IDF,
        start_idx, 
        end_idx, 
        fout, 
        lock, 
        proc_id
    ):
    start_time = time.time()
    no_constraint = 0
    n_c = 0
    n_s = 0
    for i in range(start_idx, min(len(queries), end_idx)):

        if i%100 == 0:
            print(proc_id, i-start_idx, time.time()-start_time, flush=True)
        
        # 2 queries
        query = queries[i]
        response = responses[i]
        context_sents = contexts[i]
        bestMatchedPhrasesInfo, c_res = getBestMatchedPhrase(
                                    query,
                                    context_sents,
                                    response, 
                                    word2IDF, 
                                    dataset=dataset,
                                    max_phrase_length=MAX_PHRASE_LENGTH)
        if bestMatchedPhrasesInfo is None:
            no_constraint += 1
            
        lock.acquire()
        fout.write(str(str(i)+'\n').encode("utf-8"))
        fout.write(str(query+'\n').encode("utf-8"))
        fout.write(str(response+'\n').encode("utf-8"))
        
        seen_pairs = set()
        seen_cstrs = set()
        seen_sents = set()
        if bestMatchedPhrasesInfo:
            for p_id in range(len(bestMatchedPhrasesInfo)):
                phrase = bestMatchedPhrasesInfo[p_id]
                # unfold
                best_matching_phrase = phrase[-2]
                best_matching_phrase_ctx_sent = phrase[-1]

                pair = (best_matching_phrase.lower(), best_matching_phrase_ctx_sent.lower())
                if len(best_matching_phrase) > 0 and pair not in seen_pairs:
                    fout.write(str(best_matching_phrase_ctx_sent+'\n').encode("utf-8"))
                    fout.write(str(best_matching_phrase+'\n').encode("utf-8"))
                    seen_pairs.add(pair)
                    seen_cstrs.add(best_matching_phrase)
                    seen_sents.add(best_matching_phrase_ctx_sent)
        n_c += len(seen_cstrs)
        n_s += len(seen_sents)

        fout.write('\n'.encode("utf-8"))
        lock.release()
    with_c = i-start_idx-no_constraint
    print('# examples with cstr(s) found: {}; # examples: {}; average # cstr(s) per constrained example: {}; average # sent(s) per constrained example: {}'.format(with_c, i-start_idx, n_c*1.0/with_c, n_s*1.0/with_c))

def generateQATrainingData(isTrain=True, dataset=None, numOfProcesses=10, prefix='train'):
    
    source_file = '{}/data/{}/src_tgt_ctx/{}_source.txt'.format(PROJECT_FOLDER, dataset, prefix)
    target_file = '{}/data/{}/src_tgt_ctx/{}_target.txt'.format(PROJECT_FOLDER, dataset, prefix)
    ctx_file = '{}/data/{}/src_tgt_ctx/{}_context.txt'.format(PROJECT_FOLDER, dataset, prefix)
    idf_file = '{}/prepare_data/generate_conv_with_cstr/idf_files/idf_{}.txt'.format(PROJECT_FOLDER, dataset)
    
    queries = []
    responses = []
    contexts = []

    with open(source_file) as f_src, open(target_file) as f_tgt, open(ctx_file) as f_ctx:
        for line in f_src:
            queries.append(line.strip(' \r\n'))
        for line in f_tgt:
            responses.append(line.strip(' \r\n'))
        tmp = 0
        for line in f_ctx:
            if tmp%10000 == 0:
                print(tmp)
            tmp += 1
            context_list = re.split('<[a-z|_|0-9]*>', line)[1:]
            context_name_list = [n for n in re.findall('<[a-z|_|0-9]*>', line)]
            assert len(context_name_list) == len(context_list)
    
            context_list = list(zip(context_name_list, [c.strip(' \r\n') for c in context_list]))
            contexts.append(' '.join(ctx) for ctx in context_list)
    
    assert len(queries) == len(responses) 
    assert len(queries) == len(contexts)
    instancesPerProc = int(math.ceil(len(queries)*1.0/numOfProcesses))
    
    word2IDF = loadWord2IDF(idf_file)
    print('total # queries: {}\n# processes: {}\n# queries per processes: {}'.format(
                                                               len(queries), numOfProcesses, instancesPerProc))

    # creating training data for QA andf write to output file
    lock = Lock()
    processes = []
    fout = open('{}/data/dstc/conv_with_cstr/constraints_{}.txt'.format(PROJECT_FOLDER, prefix), 'wb', buffering=0)
    
    for proc_id in range(numOfProcesses):
        start_idx = proc_id * instancesPerProc
        end_idx = (proc_id+1) * instancesPerProc
        args = (dataset, queries, responses, contexts, word2IDF, start_idx, end_idx, fout, lock, proc_id)
        p = Process(target=writeMatchedPhrasesToFile, args=args)
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    fout.close()


for prefix in PREFICES:
    generateQATrainingData(isTrain=False, dataset=DATASET, numOfProcesses=N_THREADS, prefix=prefix)
