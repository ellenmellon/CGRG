import sys
import torch
import tqdm
import logging


import numpy as np

from collections import OrderedDict
from pdb import set_trace as bp
from collections import defaultdict

from gpt2_training.generation import generate_sequence, cut_seq_to_eos, beam_search_naive

logger = logging.getLogger(__name__)

EOS_ID = 50256


def cal_entropy(generated):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score



def eval_model_loss(model, eval_dataloader, epoch_id, args, encoder=None, return_loss=False, setting='', outfile=''):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    fout = None
    if outfile != '':
        fout = open(outfile, 'a+')
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, attn_masks = batch

            if args.no_token_id:
                token_ids = None
            if args.no_attn_mask:
                attn_masks = None
            n_sample = input_ids.shape[0]

            if return_loss:
                token_level_loss, loss, ppl = model(input_ids, position_ids, token_ids, label_ids, attn_masks, None, return_loss)
            else:
                loss, ppl = model(input_ids, position_ids, token_ids, label_ids, attn_masks, None, return_loss)
            if encoder is not None:
                if fout:
                    fout.write(setting+'\n')
                    fout.write('|'.join([encoder.decode([voc_id.tolist()]) for voc_id in torch.masked_select(label_ids, label_ids.ge(0.0)).view(-1)])+'\n')
            if return_loss:
                if fout:
                    # log probabilities
                    fout.write('|'.join(['-'+str(t_loss.tolist()) for t_loss in token_level_loss.view(-1)])+'\n')
            if fout:
                # response level loss
                fout.write('{}\n\n'.format(loss.item()))

            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    if fout:
        fout.close()
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)
