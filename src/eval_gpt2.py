'''
 * @Date: 2019-04-01 13:17:00
 * @Last Modified by:   Yizhe Zhang
 * @Last Modified time: 2019-04-01 13:17:00
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
 
 This script is intended for made-up examples. It outputs both token-level and turn-level prob/loss
 to output file, which will be read by the probability plot script in ../eval/
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
parser.add_argument("--fp16", type=boolean_string, default=True)
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
parser.add_argument("--input_setting", type=str,
                    help='indicate the setting of input/model, specifically, no_cstr, ground_only, cstr_only or ground_cstr, etc')
parser.add_argument("--output_file", type=str, default='/data2/ellen/cstr_grounded_conv/src/case_studies_@/gen_loss.txt', 
                    help='output file for writing loss/probability. Better have the path to be some file under ../eval')

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')

parser.add_argument('--config', help='JSON config file')


# do normal parsing
args = parser.parse_args()

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

if not args.skip_eval:
    eval_dataloader_loss = DynamicBatchingLoader(
        args.eval_input_file, enc, args.normalize_data,
        args.eval_batch_size, args.max_seq_length,
        is_train=True)

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

#########################################################################
# Training !
##########################################################################
epoch = 0

if not args.skip_eval:
    eval_loss, eval_ppl = eval_model_loss(model, eval_dataloader_loss, epoch, args, encoder=enc, return_loss=True, setting=args.input_setting, outfile=args.output_file)
