'''
 * @Date: 2019-04-01 13:17:00
 * @Last Modified by:   Ellen Wu
 * @Last Modified time: 2021-05-25 13:17:00
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
'''
import json
import os
from os.path import abspath, dirname, exists, join
import sys
import argparse
import logging
import math
import time

import tqdm
import datetime
import numpy as np
import torch
from torch.utils.data import RandomSampler, TensorDataset, DataLoader
from torch.distributed import get_rank, get_world_size

from gpt2_training.train_utils import (
    load_model, DynamicBatchingLoader, boolean_string, set_lr,
    get_eval_list_same_length, get_len_mapper)
from gpt2_training.eval_utils import eval_model_loss

from data_loader import BucketingDataLoader, DistributedBucketingDataLoader
from env import PROJECT_FOLDER

from optim import Adam
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from gpt2_training.distributed import (all_reduce_and_rescale_tensors,
                                       all_gather_list)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=1024,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=10000,
                    help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", action='store_true')
parser.add_argument("--no_attn_mask", action='store_true')

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--save_step", type=int, default=30000)
parser.add_argument('--pbar', action='store_true', help='turn on progress bar')

# generation
parser.add_argument("--nsamples", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--length", type=int, default=-1)

parser.add_argument("--generation_length", type=int, default=20)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument('--unconditional', action='store_true',
                    help='If true, unconditional generation.')
parser.add_argument('--is_sampling', action='store_true',
                    help='If true, sampling for generation.')

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')
parser.add_argument('--gpu', type=str, default='-1')

parser.add_argument('--config', help='JSON config file')


# do normal parsing
args = parser.parse_args()

if args.gpu != '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# TODO there might be a better way to write this...
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
    setattr(args, 'local_rank', overrides.local_rank)


assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
    'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size
                         // args.gradient_accumulation_steps)
logger.info('train batch size = {}, '
            'new train batch size (after gradient accumulation) = {}'.format(
                args.train_batch_size*args.gradient_accumulation_steps,
                args.train_batch_size))


if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    print(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(args.output_dir,
                  'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

if args.fp16:
    config = join(abspath(PROJECT_FOLDER),
                  'config_file/SeqLen_vs_BatchSize_1GPU_fp16.csv')
else:
    config = join(abspath(PROJECT_FOLDER),
                  'config_file/SeqLen_vs_BatchSize_1GPU_fp32.csv')
seq_len_mapper = get_len_mapper(config)
#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))

if args.local_rank == -1:
    train_dataloader = BucketingDataLoader(args.train_input_file,
                                           args.train_batch_size,
                                           args.max_seq_length)
else:
    train_dataloader = DistributedBucketingDataLoader(
        get_rank(), get_world_size(),
        args.train_input_file, args.train_batch_size,
        args.max_seq_length)


#logger.info("***** For training dataset *****")

if not args.skip_eval:
    eval_dataloader_loss = DynamicBatchingLoader(
        args.eval_input_file, enc, args.normalize_data,
        args.eval_batch_size, args.max_seq_length,
        is_train=True)

    """
    eval_dataloader_gen = get_eval_list_same_length(
        args.eval_input_file, enc, args.eval_batch_size, True)


    logger.info("***** For dev dataset *****")
    logger.info('num example = %d, batch_size = %d, num_batches = %d'
                % (eval_dataloader_loss.num_examples, args.eval_batch_size,
                len(eval_dataloader_gen)))
    """
#########################################################################
# Prepare Model and Optimizer
##########################################################################
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                   args, verbose=True)
if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))
    # FIXME is averaging the best way? init variance will change

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
                                   verbose=False)
    else:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   verbose=False)
else:
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,
                     max_grad_norm=1.0)

#########################################################################
# Training !
##########################################################################

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=1)
    eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
    print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,eval_loss,eval_ppl', file=eval_logger)

global_step = 0
step = 0
epoch = 0
if args.continue_from:
    global_step = args.continue_from
    step = global_step*2 - 1

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None
while True:
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps
     ) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        # activate new training mode
        seq_len = batch[0].shape[1]
        max_bs = int(seq_len_mapper[seq_len]) * n_gpu
        tmp_dataset = TensorDataset(*batch)
        tmp_dataloader = DataLoader(tmp_dataset,
                                    sampler=RandomSampler(tmp_dataset),
                                    batch_size=max_bs)
        gas = len(tmp_dataloader)
        for _, tmp_batch in enumerate(tmp_dataloader):
            tmp_batch = tuple(t.to(device) for t in tmp_batch)
            input_ids, position_ids, token_ids, label_ids, attn_masks, *_ = tmp_batch
            if args.no_token_id:
                token_ids = None
            if args.no_attn_mask:
                attn_masks = None
            loss, ppl = model(input_ids, position_ids, token_ids, label_ids, attn_masks)

            if n_gpu > 1:
                loss = loss.mean()
                ppl = ppl.mean()
            if math.isnan(loss.item()):
                # skip mini-batch if NaN
                print('NaN!!', file=train_logger)
                if not exists(f'{output_dir}/debug_nan_{args.local_rank}.pt'):
                    # save snapshot for debugging
                    debug_snapshot = {
                        'state_dict': {k: v.cpu()
                                       for k, v in model.state_dict().items()},
                        'input_ids': input_ids.cpu(),
                        'position_ids': position_ids.cpu(),
                        'token_ids': (token_ids.cpu() if token_ids is not None else None),
                        'label_ids': label_ids.cpu(),
                        'attn_masks': attn_masks.cpu()}
                    torch.save(debug_snapshot,
                               f'{output_dir}/debug_nan_{args.local_rank}.pt')
                continue
            loss = loss / (args.train_batch_size / input_ids.shape[0])
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += float(loss.item()) * (
                args.train_batch_size / input_ids.shape[0])
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss / nb_tr_steps
            if ppl.item() < INF:
                tr_ppl += ppl.item()
            else:
                tr_ppl += mean_ppl
            mean_ppl = tr_ppl / nb_tr_steps

            n_token_total += input_ids.shape[0] * input_ids.shape[1]
            n_token_real += (input_ids != 0).sum().item()

        # gradient update
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,
                   args.lr_schedule, args.learning_rate,
                   args.warmup_steps, args.warmup_proportion,
                   config.n_embd, args.num_optim_steps)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Print log info to file
            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                        f"ppl: {mean_ppl:.2f} epoch: {epoch}")
                    pbar.update(1)
                print('{},{},{},{},{},{},{},{}'.format(
                    epoch+1, global_step+1, step+1, mean_loss, mean_ppl,
                    n_token_real_all_proc, n_token_total_all_proc, epoch_time),
                    file=train_logger)

            if (global_step-1) % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    if (global_step-1) % args.save_step == 0:
                        torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                             for k, v in model.state_dict().items()},
                            join(output_dir,
                                 f'GP2-pretrain-step-{global_step}.pkl'))
                    eval_loss, eval_ppl = 'n/a', 'n/a'
                    if not args.skip_eval:
                        eval_loss, eval_ppl = eval_model_loss(
                            model, eval_dataloader_loss, epoch, args)

                    print('{},{},{},{},{}'.format(
                        epoch+1, global_step+1, step+1, eval_loss, eval_ppl),
                        file=eval_logger)
                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    model.train()
            if global_step >= args.num_optim_steps:
                break

        if (step+1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if global_step >= args.num_optim_steps:
        break
    epoch += 1


if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    eval_logger.close()
