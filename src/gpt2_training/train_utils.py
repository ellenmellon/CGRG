import os
import logging
import torch
import subprocess as sp
from collections import defaultdict
from math import ceil

from torch.nn.utils.rnn import pad_sequence
import numpy as np

from env import END_OF_TURN_TOKEN, END_OF_TEXT_TOKEN
from optim import warmup_linear, noam_decay, noamwd_decay


logger = logging.getLogger(__name__)

SEQ_LENGTH_SHRINK_PROP = 0.9


def load_model(model, checkpoint, args, verbose=False):
    n_gpu = args.n_gpu
    device = args.device
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in model_state_dict.keys()):
            logger.info('loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict, strict=False) # double check imcompatible keys

    if args.fp16:
        logger.info('in fp16, model.half() activated')
        model.half()
    model.to(device)
    if n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

class RedditExample(object):
    def __init__(self, conv_id, source, response):
        self.conv_id = conv_id
        self.source = source
        self.response = response

    def __repr__(self):
        return 'conv_id = {}\nsource = {}\nresponse = {}'.format(self.conv_id, self.source, self.response)

    def __str__(self):
        return self.__repr__()


class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids, lm_labels, context_len, response_len):
        self.conv_id = conv_id
        self.choices_features = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        self.lm_labels = lm_labels
        self.context_len = context_len
        self.response_len = response_len    # in case we need it


class InputFeatures_v4(object):
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

def _get_test_feature_from_text(conv_id, text, tokenizer, offset=0):

    src, tgt, attn_masks, types, positions = text.strip().split('\t')
    input_ids = [int(num) for num in src.strip().split()] + [tokenizer.encoder[END_OF_TEXT_TOKEN]] + [int(num) for num in tgt.strip().split()[:offset]]
    tgt = ' '.join(tgt.strip().split()[offset:])
    position_ids = [int(num) for num in positions.strip().split()[:len(input_ids)]]
    type_ids = [int(num) for num in types.strip().split()[:len(input_ids)]]
    attn_masks = [[int(c) for c in seq]+[0]*(len(input_ids)-len(seq)) for seq in attn_masks.split(',')[:len(input_ids)]]


    lm_labels = [int(num) for num in tgt.strip().split()]
    weights = [1.0]*len(input_ids)

    feature = InputFeatures_v4(conv_id, input_ids, position_ids, type_ids,
                            lm_labels, attn_masks, weights)
    
    return feature

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
    feature = InputFeatures_v4(id_, input_ids, position_ids, token_type_ids,
                            lm_labels, attn_masks, weights)
    return feature


class DynamicBatchingLoader(object):
    def __init__(self, corpus_file, tokenizer, normalize_data,
                 batch_size, max_seq_length, is_train):
        self.corpus = corpus_file
        self.toker = tokenizer
        self.norm = normalize_data
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.train = is_train
        self.num_examples = self.get_len(corpus_file)

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples/self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as corpus:
                i = 0
                while True:
                    examples = []
                    cur_bs = 0
                    while True:
                        line = next(corpus)
                        weights, inputs, attn_masks, position_ids, type_ids = _get_inputs_from_text(line, self.toker)
                        features = _make_features(-1, weights, inputs, attn_masks, position_ids, type_ids, self.toker, self.max_seq_length)
                        assert len(features) == 1

                
                        examples += features
                        i += 1
                        cur_bs += 1
                        if cur_bs >= self.bs:
                            break
                    batch = self._batch_feature(features)
                    yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)

        attn_masks = torch.zeros(input_ids.shape[0], input_ids.shape[1], input_ids.shape[1])
        for i in range(len(features)):
            f = features[i]
            # seq_l * seq_l
            f_attn_masks = pad_sequence([torch.tensor(mask, dtype=torch.float)
                               for mask in f.attn_masks],
                              batch_first=True, padding_value=0)
            attn_masks[i, :f_attn_masks.shape[0], :f_attn_masks.shape[0]] = f_attn_masks
        return (input_ids, position_ids, token_type_ids, labels, attn_masks)

    def get_len(self, corpus):
        n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                     universal_newlines=True).split()[0])
        return n_line



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_eval_list_same_length(input_file, tokenizer, max_batch_size, norm=True):
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response) :
        for tgt in tgt_all:
            if norm:
                src_line = ' '.join(src.strip().split())
                tgt_line = ' '.join(tgt.strip().split())
            else:
                src_line = src.strip()
                tgt_line = tgt.strip()
            examples.append(RedditExample(i, src_line, tgt_line))
            i += 1

    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id

        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
        position_ids = torch.stack([torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
        token_type_ids = torch.stack([torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long) for f in features],
                                                 batch_first=True, padding_value=-1)

        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        return input_ids, position_ids, token_type_ids, labels, context_len, response_len

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader


def get_eval_list_same_length_with_order(input_file, tokenizer, max_batch_size, norm=True, offset=0):
    # temporary fix, will replace get_eval_list_same_length in future
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    features = []
    for line in lines:
        feature = _get_test_feature_from_text(i, line.strip(' \r\n'), tokenizer, offset=offset)
        features += [feature]
        i += 1

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.input_ids, dtype=torch.long) for f in features])
        position_ids = torch.stack([torch.tensor(f.position_ids, dtype=torch.long) for f in features])
        token_type_ids = torch.stack([torch.tensor(f.token_type_ids, dtype=torch.long) for f in features])
        attn_masks = torch.stack([torch.tensor(f.attn_masks, dtype=torch.float) for f in features])
        labels = [f.lm_labels for f in features]

        context_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
        conv_ids = torch.tensor([torch.tensor(f.conv_id, dtype=torch.long) for f in features])

        return input_ids, position_ids, token_type_ids, attn_masks, labels, context_len, conv_ids

    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.input_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader


def set_lr(optimizer, step, schedule, lr,
           warmup_steps, warmup_proportion, n_embd, tot_steps):
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step+1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step+1, warmup_steps, n_embd)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step


def get_len_mapper(config):
    seq_len = np.genfromtxt(config, delimiter=',', skip_header=1, dtype=int)
    seq_len_mapper = np.ones(seq_len[-1][0], dtype=int) * 160
    for i in range(len(seq_len)-1):
        seq_len_mapper[seq_len[i][0]] = seq_len[i][1]
        for j in range(seq_len[i][0]+1, seq_len[i+1][0]):
            seq_len_mapper[j] = seq_len[i+1][1] * SEQ_LENGTH_SHRINK_PROP
    return seq_len_mapper
