#############
# Creates input files for GPT2 + IA model. (Note: attention masks can be exploited to perform the same way as GPT2 model)
# Each line encodes features of an example, and any examples of length > max_length will be discarded
#############

import sys
import re
import argparse
from pytorch_pretrained_bert import GPT2Tokenizer

def main(args):
    toker = GPT2Tokenizer.from_pretrained(args.tokenizer)
    mean_input_l = 0.0
    n_valid = 0
    tot_num = 0

    with open(args.cstr_fname) as fin, open(args.output_fname, 'w') as fout:
        examples = fin.read().split('\n\n')
        skip_ids = []
        eid = 0
        for e in examples:
            if args.keep_case:
                lines = e.split('\n')
            else:
                lines = [l.lower() for l in e.split('\n')]
            if len(lines) < 3:
                skip_ids += [eid]
                eid += 1
                continue
            prev_turns = re.split('<t[0-9]>', lines[1].strip(' \r\n'))
            if prev_turns[0].strip(' \r\n') == '':
                prev_turns = prev_turns[1:]
            target = toker.encode(lines[2].strip(' \r\n'))

            for i in range(len(prev_turns)):
                prev_turns[i] = prev_turns[i].strip(' \r\n') + ' <|endoftext|> '

            # convert prev_turns to list of ids (encode)
            source = []
            for i in range(len(prev_turns)):
                source.append(toker.encode(prev_turns[i]))


            idx = 3
            constraints = []
            c_sents = []
            cstr_enc = []
            sent_enc = []
            sent2idx = []
            
            cstr2idx = []
            cstr2sents =[]

            # TODO: Add option of inserting one constraint only
            while idx < len(lines) and len(constraints) < args.max_n_cstr:
                if lines[idx+1].strip(' \r\n') == 'n/a':
                    break
                c = lines[idx+1].strip(' \r\n') + ' <c> '
                s = lines[idx].strip(' \r\n') + ' <s> '
                c_enc = toker.encode(c)
                s_enc = toker.encode(s)
                if c not in constraints:
                    constraints += [c]
                    cstr_enc += [c_enc]
                    if len(cstr2idx) == 0:
                        cstr2idx += [(0, len(c_enc))]
                    else:
                        cstr2idx += [(cstr2idx[-1][1], cstr2idx[-1][1]+len(c_enc))]
                if s not in c_sents and len(c_sents) < args.max_n_sents:
                    c_sents += [s]
                    sent_enc += [s_enc]
                    if len(sent2idx) == 0:
                        sent2idx += [(0, 0+len(s_enc))]
                    else:
                        sent2idx += [(sent2idx[-1][1], sent2idx[-1][1]+len(s_enc))]

                if len(cstr2sents) < len(constraints):
                    assert len(cstr2sents) + 1 == len(constraints)
                    try:
                        cstr2sents.append([c_sents.index(s)])
                    except:
                        # s in not inserted to c_sents due to number limit
                        assert len(c_sents) == args.max_n_sents
                        cstr2sents.append([])
                else:
                    try:
                        cstr2sents[constraints.index(c)] += [c_sents.index(s)]
                    except:
                        # s in not inserted to c_sents due to number limit
                        assert len(c_sents) == args.max_n_sents

                idx += 2
            total_length = 0
            if len(constraints) > 0:
                total_length += sent2idx[-1][1] + cstr2idx[-1][1] + len(source[-1])
            else:
                total_length += len(source[-1])

            tot_num += 1

            if total_length > args.max_seq_length or len(target) > args.max_seq_length:
                skip_ids += [eid]
                eid += 1

                continue
            n_valid += 1
            eid += 1

            tid = 0
            while tid < len(source)-1:
                if total_length + len(source[-tid-2]) > args.max_seq_length:
                    break
                tid += 1

            source = source[-tid-1:]
            dh = []
            for s in source:
                dh += s
            source_end_idx = len(dh)

            for sid in range(len(sent2idx)):
                sent2idx[sid] = (sent2idx[sid][0]+source_end_idx, sent2idx[sid][1]+source_end_idx)
            for cid in range(len(cstr2idx)):
                cstr2idx[cid] = (cstr2idx[cid][0]+sent2idx[-1][1], cstr2idx[cid][1]+sent2idx[-1][1])


            # 
            # calculate attn_masks
            #

            mask_seq = [[1]*(i+1) for i in range(source_end_idx)]
            for sid in range(len(sent2idx)):
                shared = [1]*source_end_idx+[0]*(sent2idx[sid][0]-source_end_idx)
                mask_seq += [shared+[1]*(i-sent2idx[sid][0]+1) for i in range(sent2idx[sid][0], sent2idx[sid][1])]
            for cid in range(len(cstr2idx)):
                shared = [1]*source_end_idx
                for sid in range(len(sent2idx)):
                    shared += [int(sid in cstr2sents[cid])]*(sent2idx[sid][1]-sent2idx[sid][0])
                shared += [0]*(cstr2idx[cid][0]-sent2idx[-1][1])
                mask_seq += [shared+[1]*(i-cstr2idx[cid][0]+1) for i in range(cstr2idx[cid][0], cstr2idx[cid][1])]

            shared = [1]*source_end_idx
            if len(cstr2idx) > 0:
                shared += [1]*(cstr2idx[0][0]-source_end_idx) + [1]*(cstr2idx[-1][1]-cstr2idx[0][0])
            mask_seq += [shared+[1]*(i+1) for i in range(len(target)+1)] # 'eos' token at the beginning

            mask_seq_str = []
            for i in range(len(mask_seq)):
                assert len(mask_seq[i]) == i+1
                mask_seq_str += [''.join([str(j) for j in mask_seq[i]])]
            mask_seq_str = ','.join(mask_seq_str)


            #
            # calculate position_ids and type_ids
            #
            position_ids = list(range(source_end_idx))
            type_ids = [0]*source_end_idx 
            for sid in range(len(sent2idx)):
                position_ids += list(range(sent2idx[sid][1]-sent2idx[sid][0]))
                type_ids += [1+sid]*(sent2idx[sid][1]-sent2idx[sid][0])
            for cid in range(len(cstr2idx)):
                position_ids += list(range(cstr2idx[cid][1]-cstr2idx[cid][0]))
                type_ids += [1+args.max_n_sents+cid]*(cstr2idx[cid][1]-cstr2idx[cid][0])
            position_ids += list(range(len(target)+1))
            type_ids += [1+args.max_n_sents+args.max_n_cstr]*(len(target)+1)
            assert len(position_ids) == len(mask_seq)
            assert len(type_ids) == len(mask_seq)



            #
            # convert variables to strings for writing to file
            #
            source_str = []
            for i in range(len(source)):
                source_str += [' '.join([str(t) for t in source[i]])]
            source_str = ' '.join(source_str)

            sent_str = []
            for i in range(len(sent_enc)):
                sent_str += [' '.join([str(t) for t in sent_enc[i]])]
            sent_str = ' '.join(sent_str)

            cstr_str = []
            for i in range(len(cstr_enc)):
                cstr_str += [' '.join([str(t) for t in cstr_enc[i]])]
            cstr_str = ' '.join(cstr_str)

            target_str = ' '.join([str(t) for t in target])
            position_ids_str = ' '.join([str(t) for t in position_ids])
            type_ids_str = ' '.join([str(t) for t in type_ids])

            if len(sent_str) > 0:
                fout.write('{} {} {}\t{}\t{}\t{}\t{}\n'.format(source_str, sent_str, cstr_str, target_str, mask_seq_str, type_ids_str, position_ids_str))
            else:
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(source_str, target_str, mask_seq_str, type_ids_str, position_ids_str))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cstr_fname', '-c', required=True,
                        help='file name of constrained conv')
    parser.add_argument('--output_fname', '-o', required=True,
                        help='file name of output')
    parser.add_argument('--tokenizer', required=True, help='pretrained tokenizer path')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_n_sents', type=int, default=20)
    parser.add_argument('--max_n_cstr', type=int, default=10)
    parser.add_argument('--keep_case', action='store_true')
    args = parser.parse_args()
    main(args)
