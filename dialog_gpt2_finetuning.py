import os
import sys
import json
import pickle as pkl
import random
import time
import math
import pickle
import numpy as np
import logging
import argparse
import tqdm.notebook as tqdm
import time
import collections as cl
from typing import Dict, List, Tuple
from apex import amp
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup, BertModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig
SEED = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device {} ".format(device))
logger = logging.getLogger(__name__)

data_path = "data/dataset/v2/dialogue_data/context_2_20/"

train_data_c1 = []
train_data_c2 = []
with open(data_path + 'train_context_text.txt', 'r', encoding='latin1') as f:
    for line in f.readlines():
        c1, c2 = line[:-1].split('|')
        train_data_c1.append(c1)
        train_data_c2.append(c2)


train_data_r = []
with open(data_path + 'train_target_text.txt', 'r', encoding='latin1') as f:
    for line in f.readlines():
        train_data_r.append(line[:-1])

assert len(train_data_c1)==len(train_data_r)

val_data_c1 = []
val_data_c2 = []
with open(data_path + 'valid_context_text.txt', 'r', encoding='latin1') as f:
    for line in f.readlines():
        c1, c2 = line[:-1].split('|')
        val_data_c1.append(c1)
        val_data_c2.append(c2)

val_data_r = []
with open(data_path + 'valid_target_text.txt', 'r', encoding='latin1') as f:
    for line in f.readlines():
        val_data_r.append(line[:-1])

assert len(val_data_c1)==len(val_data_r)

class Args():
    def __init__(self):
        self.output_dir = 'models/output-large'
        self.model_type = 'gpt2'
        self.model_name_or_path = 'microsoft/DialoGPT-large'
        self.config_name = 'microsoft/DialoGPT-large'
        self.tokenizer_name = 'microsoft/DialoGPT-large'
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 2
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 100000000
        self.save_steps = 35000000
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'

args = Args()

config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
model = AutoModelWithLMHead.from_pretrained(
    args.model_name_or_path,
    from_tf=False,
    config=config,
    cache_dir=args.cache_dir,
)

def get_ind(c1, c2, r):
    t = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in [c1, c2, r]])
    t = [item for sublist in t for item in sublist]
    return t

class MultimodalDataset(Dataset):
    def __init__(self, tokenizer, args, data_c1, data_c2, data_r, 
                 name='train', block_size=512):

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        if(name == 'train'):
            cached_features_file = 'train_indices.pkl'
        else:
            cached_features_file = 'valid_indices.pkl'

        if os.path.exists(cached_features_file):
            print("Loading features from cached file: ", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = []
            for i in tqdm(range(len(data_c1))):
                self.examples.append(get_ind(data_c1[i], data_c2[i], data_r[i]))

            print("Saving features into cached file: ", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

train_dataset = MultimodalDataset(tokenizer, args, train_data_c1, train_data_c2,
                                 train_data_r, name='train')

#train_ind = []
#for i in tqdm.tqdm(range(len(train_data_c1))):
#    train_ind.append(get_ind(train_data_c1[i], train_data_c2[i], train_data_r[i]))

#with open('train_indices.pkl', "rb") as handle:
#    train_ind = pickle.load(handle)

#val_ind = []
#for i in tqdm.tqdm(range(len(val_data_c1))):
#    val_ind.append(get_ind(val_data_c1[i], val_data_c2[i], val_data_r[i]))

#with open('valid_indices.pkl', "rb") as handle:
#    val_ind = pickle.load(handle)

#with open('train_indices.pkl', "wb") as handle:
#    pickle.dump(train_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('valid_indices.pkl', "wb") as handle:
#    pickle.dump(val_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = model.to(device)

def add_padding(sentences):
    if tokenizer._pad_token is None:
        return pad_sequence(sentences, batch_first=True)
    return pad_sequence(sentences, batch_first=True, padding_value=tokenizer.pad_token_id)

n_gpu = torch.cuda.device_count()
train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=add_padding, drop_last = True
)

if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

model.resize_token_embeddings(len(tokenizer))


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
)


if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

# multi-gpu training (should be after apex fp16 initialization)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    
if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

global_step = 0
epochs_trained = 0
steps_trained_in_current_epoch = 0
# Check if continuing training from a checkpoint
if args.model_name_or_path and os.path.exists(args.model_name_or_path):
    try:
        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    except ValueError:
        logger.info("  Starting fine-tuning.")

tr_loss, logging_loss = 0.0, 0.0

model.zero_grad()
train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
)

set_seed(args)  # Added here for reproducibility
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        inputs, labels = (batch, batch)
        if inputs.shape[1] > 1024: continue
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.train()
        outputs = model(inputs, labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if (
                    args.local_rank == -1 and args.evaluate_during_training
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = "checkpoint"
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                _rotate_checkpoints(args, checkpoint_prefix)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    if args.max_steps > 0 and global_step > args.max_steps:
        train_iterator.close()
        break

if args.do_train:
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    print("Saving model checkpoint to: ", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelWithLMHead.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model.to(device)


eval_dataset = MultimodalDataset(tokenizer, args, val_data_c1, val_data_c2
                                 val_data_r, name='valid')

checkpoints = args.output_dir

model = AutoModelWithLMHead.from_pretrained(checkpoints)
model = model.to(device)

eval_output_dir = args.output_dir

os.makedirs(eval_output_dir, exist_ok=True)
eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)

eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(
    eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=add_padding, drop_last = True
)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Eval!
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_dataset))
logger.info("  Batch size = %d", eval_batch_size)
eval_loss = 0.0
nb_eval_steps = 0
model.eval()

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    inputs, labels = (batch, batch)
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs, labels=labels)
        lm_loss = outputs[0]
        eval_loss += lm_loss.mean().item()
    nb_eval_steps += 1

eval_loss = eval_loss / nb_eval_steps
perplexity = torch.exp(torch.tensor(eval_loss))

result = {"perplexity": perplexity}

print("***** Eval results *****")
for key in sorted(result.keys()):
    print("%s = %s\n" % (key, str(result[key].item())))
