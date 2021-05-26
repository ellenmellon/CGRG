"""
generate files in src_tgt_ctx for dstc_grounded
change paths before running

"""
import sys
from collections import defaultdict
import os
import spacy

# path for unprocessed dataset folder that contains 
# files of converstaions and grounding documents
DATA_FOLDER = '../data/dstc/raw/'
OUTPUT_FOLDER = '../data/dstc/src_tgt_ctx'

SUFFICES = ['train', 'dev', 'test']
USE_LOWERCASE = False



# right now, it's assuming each line contains ONLY ONE target response
def read_conv_turns_from_line(line, use_lowercase):
    content = line.strip('\r\n').split('\t')
    thread_id = content[2]
    turns = content[-2].split(' EOS ')
    final_turn = content[-1]
    turns.append(final_turn)

    clean_turns = []
    for t in turns:
        if t.strip() == '' or t == 'START' or t == '...':
            continue
        if use_lowercase:
            t = t.lower()
        clean_turns += [t.strip()]
    return clean_turns


def get_valid_fact(fact):
    tokens = fact.strip(' \r\n').split()
    if len(tokens) < 2:
        return None
    if tokens[0].startswith('<') and tokens[0].endswith('>'):
        html_tag = tokens[0][1:-1]
        if (tokens[-1] ==  '</' + html_tag + '>') and len(tokens) > 2:
            return ' '.join(tokens[:-1])
    return None





for suffix in SUFFICES:
    folder = DATA_FOLDER + suffix

    n_processed_files = 0
    f_s = open(os.path.join(OUTPUT_FOLDER, '{}_source.txt'.format(suffix)), 'w')
    f_t = open(os.path.join(OUTPUT_FOLDER, '{}_target.txt'.format(suffix)), 'w')
    f_c = open(os.path.join(OUTPUT_FOLDER, '{}_context.txt'.format(suffix)), 'w')

    for file in os.listdir(folder):        
        if (file.endswith("convos.txt") and suffix != 'test') or (file.endswith("refs.txt") and suffix == 'test'):
            print('processed {} files'.format(n_processed_files))
            n_processed_files += 1
            cid_2_turns = {}
            cid_2_thread_id = {}
            thread_id_2_ctx = defaultdict(list)
            cid = 0
    
            conv_file = os.path.join(folder, file)
            if suffix != 'test':
                fact_file = conv_file.replace('convos', 'facts')
            else:
                fact_file = conv_file.replace('refs', 'facts')
    
            ct=0
            with open(conv_file) as f_conv:
                for line in f_conv:
                    if ct%1000==0:
                        print(ct)
                    ct += 1
                    clean_turns = read_conv_turns_from_line(line, USE_LOWERCASE)
                    if len(clean_turns) > 1:
                        cid_2_turns[cid] = clean_turns
                        content = line.strip('\r\n').split('\t')
                        thread_id = content[2]
                        cid_2_thread_id[cid] = thread_id
                        cid += 1
    
            with open(fact_file) as f_fact:
                for line in f_fact:
                    content = line.strip('\r\n').split('\t')
                    thread_id = content[2]
                    fact = content[-1]
                    valid_fact = get_valid_fact(fact)
                    if not valid_fact:
                        continue
                    if USE_LOWERCASE:
                        valid_fact = valid_fact.lower()
                    if valid_fact:
                        thread_id_2_ctx[thread_id] += [valid_fact]
            
            for cid in cid_2_turns:
                turns = cid_2_turns[cid]
                ctx = ' '.join(thread_id_2_ctx[cid_2_thread_id[cid]])
                for turn_id in range(len(turns)-1, len(turns)):
                    tgt = turns[turn_id]
                    src = turns[max(0, turn_id-3):turn_id]
                    src = ' '.join(['<t{}> {}'.format(i+1, src[i]) for i in range(len(src))])
                    f_s.write(src+'\n')
                    f_t.write(tgt+'\n')
                    f_c.write(ctx+'\n')
