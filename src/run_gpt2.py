#!/usr/bin/env python3
# NOTE: use beam search and beam size = 1 only. (This is consistent with the paper)
# The beam search is not working for beam_size > 1 now.

import json
from os.path import abspath, dirname, exists, join
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket
import os, sys
import re

from env import PROJECT_FOLDER

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from gpt2_training.train_utils import load_model, DynamicBatchingLoader, boolean_string, get_eval_list_same_length_with_order
from gpt2_training.generation import generate_sequence, cut_seq_to_eos, beam_search_naive
from gpt2_training.eval_utils import EOS_ID


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def run_model():
    print(socket.gethostname())

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/philly/sc3/resrchvc/yizzhang/GPT/pretrained/117M', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='/philly/sc3/resrchvc/yizzhang/GPT/pretrained/117M/pytorch_model.bin')
    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--test_file", '-t', type=str, default=None, help='input file for testing')
    parser.add_argument("--output_file", '-o', type=str, default=None, help='output file for testing')
    parser.add_argument("--normalize_data", type=boolean_string, default=True)
    parser.add_argument("--batch_size", '-b', type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--no_token_id", action='store_true')
    parser.add_argument("--no_attn_mask", action='store_true')
    parser.add_argument("--no_eos", action='store_true')
    
    parser.add_argument("--generation_length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--is_sampling', action='store_true', help='If true, sampling for generation.')
    parser.add_argument('--output_ref', action='store_true', help='If true, output ref')

    #BEAM
    parser.add_argument("--beam", action='store_true', help='If true, beam search')
    parser.add_argument("--beam_width", type=int, default=1)
    
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--config', help='JSON config file')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cstr_decode', action='store_true')
    parser.add_argument("--bonus", type=float, default=0.0)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    if args.config is not None:
        # override argparse defaults by config JSON
        opts = json.load(open(args.config))
        for k, v in opts.items():
            if isinstance(v, str):
                # PHILLY ENV special cases
                if 'PHILLY_JOB_DIRECTORY' in v:
                    v = v.replace('PHILLY_JOB_DIRECTORY',
                                os.environ['PHILLY_JOB_DIRECTORY'])
                elif 'PHILLY_LOG_DIRECTORY' in v:
                    v = v.replace('PHILLY_LOG_DIRECTORY',
                                os.environ['PHILLY_LOG_DIRECTORY'])
            setattr(args, k, v)

        # command line should override config JSON
        argv = sys.argv[1:]
        overrides, _ = parser.parse_known_args(argv)
        for k, v in vars(overrides).items():
            if f'--{k}' in argv:
                setattr(args, k, v)
        # setattr(args, 'local_rank', overrides.local_rank)

# do normal parsing


    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
    print(args)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args, verbose=True)
    model.to(device)
    model.eval()


    if args.test_file:
        eval_dataloader = get_eval_list_same_length_with_order(args.test_file, enc, args.batch_size, True)

    
        model.eval()
        outs = []
        targets = []
        loss_all = []
        ppl_all = []
        sources = []
        conv_ids = []
        with torch.no_grad():
            with tqdm.tqdm(total=len(eval_dataloader), desc=f"Test") as pbar:
                for step, batch in enumerate(tqdm.tqdm(eval_dataloader, desc="Iteration")):               

                    new_batch = []
                    for t in batch:
                        if isinstance(t,list):
                            new_batch.append(t)
                        else:
                            new_batch.append(t.to(device))

                    input_ids, position_ids, token_ids, attn_masks, label_ids, context_len, conv_id = new_batch  

                    if args.no_token_id:
                        token_ids = None
                    if args.no_eos:
                        input_ids = input_ids[:,:-1]
                    if args.no_attn_mask:
                        attn_masks = None
                    if args.beam:
                        out = beam_search_naive(model, input_ids, position_ids=position_ids, token_type_ids=token_ids, 
                                                attn_masks=attn_masks,
                                                length=args.generation_length,
                                                beam_width=args.beam_width, device=args.device, use_bonus=args.cstr_decode, bonus=args.bonus, enc=enc)
                    else:
                        out = generate_sequence(model, input_ids, position_ids=position_ids, token_type_ids=token_ids, 
                                        attn_masks=attn_masks,
                                        length=args.generation_length,
                                        start_token=None,
                                        temperature=args.temperature, top_k=args.top_k,
                                        sample=args.is_sampling, use_bonus=args.cstr_decode, bonus=args.bonus, enc=enc
                                        )

                    sources.extend(input_ids.cpu().numpy())
                    out = out.tolist()
                    outs.extend(out)
                    targets.extend(label_ids)
                    conv_ids.extend(conv_id.cpu().numpy())

                conv_id_map = {conv_ids[i]: i for i in range(len(conv_ids))}
                val_src = [enc.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8') for s in sources]
                #print(len(val_src),len(targets))

                val_set = [enc.decode(s).encode('utf-8').decode('utf-8') for s in targets]
                gen = [enc.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8') for s in outs]

                val_src_orders = [val_src[conv_id_map[i]] for i in sorted(conv_id_map)]
                val_set_orders = [val_set[conv_id_map[i]] for i in sorted(conv_id_map)]
                gen_orders = [gen[conv_id_map[i]] for i in sorted(conv_id_map)]

                print("=" * 40 + " SAMPLE " + "=" * 40)
                src = enc.decode([x for x in input_ids[-1].cpu().numpy() if x != 0]).encode('utf-8').decode('utf-8')
                gt = val_set[-1]
                resp = gen[-1]
                print(f"Source: \t {src} \n Oracle: \t {gt} \n Resp: \t {resp}\n")
                if args.output_file:
                    with open(args.test_file[:-3] + args.output_file + '.resp.txt', "w") as resp_f:
                        for i,r in enumerate(gen_orders):
                            r = re.sub("\n", "", r)
                            if args.output_ref:
                                # import pdb; pdb.set_trace()
                                resp_f.write(val_src_orders[i] + '\t' + val_set_orders[i] + '\t' + r + '\n')
                            else:
                                resp_f.write(r + '\n')
                print("="*80)

                sys.stdout.flush()

    else:
        generated = 0
        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text) + [EOS_ID]
            context_tokens = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)#.repeat(batch_size, 1)
            generated += 1
            position_ids = torch.arange(0, context_tokens.size(-1), dtype=torch.long, device=context_tokens.device)
            token_ids = None if args.no_token_id else torch.zeros_like(context_tokens, dtype=torch.long, device=context_tokens.device)
            if args.beam:
                out = beam_search_naive(model, context_tokens, position_ids=None, token_type_ids=token_ids, 
                        length=args.generation_length,
                        beam_width=args.beam_width, device=args.device)
            else:
                out = generate_sequence(model, context_tokens, position_ids=None, token_type_ids=token_ids, 
                                        length=args.generation_length,
                                        start_token=None,
                                        temperature=args.temperature, top_k=args.top_k,
                                        sample=args.is_sampling)
            out = out.tolist()                        
            text = enc.decode(cut_seq_to_eos(out[0])).encode('utf-8').decode('utf-8')
            print("=" * 40 + " RESPONSE " + str(generated) + " " + "=" * 40)
            print(text)
            print("=" * 80)


if __name__ == '__main__':
    run_model()

