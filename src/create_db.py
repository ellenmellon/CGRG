"""
preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
from pytorch_pretrained_bert import GPT2Tokenizer
from tqdm import tqdm

from env import END_OF_TEXT_TOKEN


class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 lm_labels, attn_masks, weights, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.attn_masks = attn_masks
        self.weights = weights
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


def _get_file_len(corpus):
    print(f"wc -l {corpus}")
    n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                 universal_newlines=True).split()[0])
    return n_line


def _norm_text(text):
    toks = text.strip().split()
    w = 1.0
    return w, ' '.join(toks)


def _get_inputs_from_text(text, tokenizer):
    srcs, tgt, attn_masks, types, positions = text.strip().split('\t')

    weights = []
    inputs = []
    for src in srcs.split(' EOS '):
        context_id = [int(num) for num in src.strip().split()]
        weights.append(1.0)
        inputs.append(context_id)
    response_id = [int(num) for num in tgt.strip().split()]
    weights.append(1.0)
    inputs.append(response_id)

    attn_masks = [[int(c) for c in seq] for seq in attn_masks.split(',')]
    position_ids = [int(num) for num in positions.strip().split()]
    type_ids = [int(num) for num in types.strip().split()]
    return weights, inputs, attn_masks, position_ids, type_ids


def _make_features(id_, weights, inputs, attn_masks, position_ids, type_ids, tokenizer, max_len):
    end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
    features = []
    sents = []
    ws = []
    len_ = 0
    i = 0
    for ids, w in zip(inputs, weights):
        sents.append(ids)
        ws.append(w)
    assert len(sents) == 2
    feat = _make_feature(id_, sents, ws, attn_masks, position_ids, type_ids, end_of_text_id)
    if feat is not None:
        features.append(feat)

    return features


def _make_feature(id_, sents, ws, attn_masks, position_ids, token_type_ids, eos):
    if all(w == 0 for w in ws[1:]):
        return None
    input_ids = [i for s in sents for i in s+[eos]][:-1]
    lm_labels = []
    weights = []
    #token_type_ids = []  # this becomes round ids
    for i, (s, w) in enumerate(zip(sents, ws)):
        if i == 0:
            lm_labels += [-1] * len(s)
            weights += [0.0] * len(s)
            continue

        if w == 0.0:
            lm_labels += [-1] * (len(s) + 1)
            weights += [0.0] * (len(s) + 1)
        else:
            lm_labels += (s + [eos])
            weights += [w] * (len(s) + 1)

    # handle trailing -1's
    i = len(lm_labels) - 1
    while i >= 0:
        if lm_labels[i] != -1:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    weights = weights[:i+1]
    token_type_ids = token_type_ids[:i+1]
    attn_masks = attn_masks[:i+1]
    position_ids = position_ids[:i+1]
    token_type_ids = token_type_ids[:i+1]

    # pad to multiples of 8
    while len(input_ids) % 8 != 0:
        input_ids.append(0)
        token_type_ids.append(0)
        lm_labels.append(-1)
        weights.append(0.0)
        attn_masks.append([0]*(len(attn_masks)+1))
        position_ids.append(0)

    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(weights) == len(attn_masks))
    assert len(input_ids) % 8 == 0
    if len(input_ids) == 0:
        import ipdb
        ipdb.set_trace()
    feature = InputFeatures(id_, input_ids, position_ids, token_type_ids,
                            lm_labels, attn_masks, weights)
    return feature


def main(args):
    # TODO specify vocab path to avoid download
    if args.tokenizer is not None:
        toker = GPT2Tokenizer.from_pretrained(args.tokenizer)
    else:
        toker = GPT2Tokenizer.from_pretrained('gpt2')
    assert args.corpus.endswith('.txt') or args.corpus.endswith('.tsv')
    db_path = f'{args.corpus[:-4]}.db/db'
    if exists(dirname(db_path)):
        raise ValueError('Found existing DB, please backup')
    else:
        os.makedirs(dirname(db_path))
    with open(args.corpus, "r", encoding="utf-8") as reader, \
            shelve.open(db_path, 'n') as db:
        chunk = []
        n_chunk = 0
        n_example = 0
        for line in tqdm(reader, total=_get_file_len(args.corpus)):
            try:
                if len(chunk) >= args.chunk_size:
                    # save and renew chunk
                    db[f'chunk_{n_chunk}'] = gzip.compress(
                        json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
                    chunk = chunk[args.chunk_size:]
                    n_chunk += 1

                weights, inputs, attn_masks, position_ids, type_ids = _get_inputs_from_text(line, toker)
                if len(weights) < 2:
                    continue
                features = _make_features(n_example, weights, inputs, attn_masks, position_ids, type_ids, toker, args.max_seq_len)
                for feature in features:
                    chunk.append(vars(feature))
                    n_example += 1
            except Exception as e:
                raise e

        # save last chunk
        db[f'chunk_{n_chunk}'] = gzip.compress(
            json.dumps(chunk).encode('utf-8'))
    # save relevant information to reproduce
    meta = {'n_example': n_example,
            'chunk_size': args.chunk_size,
            'max_seq_len': args.max_seq_len}
    with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
        json.dump(meta, writer, indent=4)
    torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='file name of training corpus (should be .tsv)')
    parser.add_argument('--chunk_size', type=int, default=65536,
                        help='num of data examples in a storing chunk')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='discard data longer than this')
    parser.add_argument('--tokenizer', help='pretrained tokenizer path')

    args = parser.parse_args()

    main(args)
