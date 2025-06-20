import argparse
import os
import sys
import logging
import pickle
from functools import partial
import time
from tqdm import tqdm
from collections import Counter
import random
import numpy as np
import time
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor

from transformers import AdamW, T5Tokenizer
from t5 import MyT5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, accuracy_score

from data_utils import ABSADataset, task_data_list
from const import *
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores, extract_spans_para
import copy

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    #print(f"Random seed set as {seed}")


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument(
        "--task",
        default='asqp',
        choices=["asqp", "acos", "memd"],
        type=str,
        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument(
        "--first_stage_views",
        default='exclude_O',
        choices=["exclude_A", "exclude_C", "exclude_O", "exclude_S", "asqp", "acos", "memd"],
        type=str,
        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument(
        "--dataset",
        default='rest15',
        type=str,
        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument(
        "--eval_data_split",
        default='test',
        choices=["test", "dev"],
        type=str,
    )
    parser.add_argument("--model_name",
                        default='t5-base',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir",
                        default='outputs/temp',
                        type=str,
                        help="Output directory")
    parser.add_argument("--load_ckpt_name",
                        default=None,
                        type=str,
                        help="load ckpt path")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument(
        "--do_inference",
        default=True,
        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=25,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--load_path_cache",
                        action='store_true',
                        help="load decoded path from cache")
    parser.add_argument("--sort_label",
                        action='store_true',
                        help="sort tuple by order of appearance")
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--constrained_decode",
                        action="store_true",
                        help='constrained decoding when evaluating')

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return args


class T5FineTuner_1(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        #print("evaluating f1...")
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        scores, _, _ = compute_scores(dec, target, verbose=False)
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    task_name=args.task,
                                    data_name=args.dataset,
                                    data_type="train",
                                    n_tuples=0, #dummy
                                    wave=1,
                                    args=self.config,
                                    max_len=self.config.max_seq_length)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=10)
        
        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  task_name=args.task,
                                  data_name=args.dataset,
                                  data_type="dev",
                                  n_tuples=0,
                                  wave=1,
                                  args=self.config,
                                  max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=10)

    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                 input_ids):
        if not os.path.exists('./force_tokens.json'):
            dic = {'special_tokens':[]}
            special_tokens_tokenize_res = []
            for w in ['[O','[A','[S','[C','[SS']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]
            # 784는 '['를 의미
            dic['special_tokens'] = special_tokens_tokenize_res
            import json
            with open("force_tokens.json", 'w') as f:
                json.dump(dic, f, indent=4)
        
        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
            # ] 가 존재하면 그것의 마지막 index, 없으면 -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        # [ 가 존재하면 그것의 마지막 index, 없으면 -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens['special_tokens']
            # [ 이후에 A, C, O, S만 오도록 강요
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
            # A, C, O, S 이후에 ]만 오도록 강요
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 
            # SS 이후에 EP만 오도록 강요
        

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
            # 생성 첫번째에는 무조건 [ 만 오도록 강요
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
            # [A 이후에는, ] 만 오도록, A가 오는 것은 앞에서 이미 처리됨.
        else:
            cur_term = input_ids[last_left_brace_pos + 1]
            # 가장 마지막 sentiment term
         
        # 이 아래를 많이 바꾸는 게 좋음
        ret = []
        
        if cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
            # SS 이후에는 ...?
            
        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            # '[' 전에 ']'가 먼저 오면 안되겠지?
            for w in force_tokens['special_tokens']:
                ret.discard(w)
                # []가 다 있는데, special token이 오는건 아니겠지?
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
            # 후보군에 '['가 더 많으면, ']'를 추가
        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        # 후보군에 [ 추가
        # print("prefix_allowed_tokens_fn: ", ret[0])
        return ret
    

class T5FineTuner_2(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        #print("evaluating f1...")
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        print("dec: ", dec[0])
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        scores, _, _ = compute_scores(dec, target, verbose=False)
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    task_name=args.task,
                                    data_name=args.dataset,
                                    data_type="train",
                                    n_tuples=0,
                                    wave=2,
                                    args=self.config,
                                    max_len=self.config.max_seq_length)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=10)
        
        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  task_name=args.task,
                                  data_name=args.dataset,
                                  data_type="dev",
                                  n_tuples=0,
                                  wave=2,
                                  args=self.config,
                                  max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=10)

    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                 input_ids):

        if not os.path.exists('./force_tokens_full.json'):
            dic = {"cate_tokens":{}, "all_tokens":{}, "sentiment_tokens":{}, 'special_tokens':[]}
            for task in force_words.keys():
                dic["all_tokens"][task] = {}
                for dataset in force_words[task].keys():
                    cur_list = force_words[task][dataset]
                    tokenize_res = []
                    for w in cur_list:
                        tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0])
                    dic["all_tokens"][task][dataset] = tokenize_res
                    # all_tokens = {task: {dataset: cate + paraphrased sentiment + [SSEP], ...}, ...}
            for k,v in cate_list.items():
                tokenize_res = []
                for w in v:
                    tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
                dic["cate_tokens"][k] = tokenize_res
                # cate_tokens = {dataset: cate, ...}
            sp_tokenize_res = []
            for sp in ['great', 'ok', 'bad']:
                sp_tokenize_res.extend(self.tokenizer(sp, return_tensors='pt')['input_ids'].tolist()[0])
            for task in force_words.keys():
                dic['sentiment_tokens'][task] = sp_tokenize_res
            # sentiment_tokens = sp_tokenize_res
            special_tokens_tokenize_res = []
            for w in ['[O','[A','[S','[C','[SS']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]
            # 784는 '['를 의미
            dic['special_tokens'] = special_tokens_tokenize_res
            import json
            with open("force_tokens_full.json", 'w') as f:
                json.dump(dic, f, indent=4)
        
        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
            'it': [34],
            'null': [206,195]
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens_full['special_tokens']
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['EP'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
        else:
            cur_term = input_ids[last_left_brace_pos + 1]

        ret = []
        if cur_term in to_id['SP']:  # SP
            ret = force_tokens_full['sentiment_tokens'][task]
        elif cur_term in to_id['AT']:  # AT
            force_list = source_ids[batch_id].tolist()
            if task != 'aste':  
                force_list.extend(to_id['it'] + [1])  
            ret = force_list  
        elif cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
        elif cur_term in to_id['AC']:  # AC
            ret = force_tokens_full['cate_tokens']['laptop16'] if data_name in ['laptop_MEMD'] else force_tokens_full['cate_tokens'][data_name]
        elif cur_term in to_id['OT']:  # OT
            force_list = source_ids[batch_id].tolist()
            if task == "acos":
                force_list.extend(to_id['null'])  # null
            ret = force_list
        else:
            raise ValueError(cur_term)

        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            for w in force_tokens_full['special_tokens']:
                ret.discard(w)
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        return ret


def evaluate(model1, model2, task, data, data_type):
    """
    Compute scores given the predictions and gold labels
    """
    tasks, datas, sents, _ = read_line_examples_from_file(
        f'../data/{task}/{data}/{data_type}.txt', data_type, task, data, lowercase=False)
    
    orders, order_targets, outputs, targets, probs = [], [], [], [], []
    cache_file = os.path.join(
        args.output_dir, "result_{}{}{}_{}_path{}_beam{}.pickle".format(
            "best_" if args.load_ckpt_name else "",
            "cd_" if args.constrained_decode else "", task, data, 5,
            args.beam_size))
    if args.load_path_cache:
        with open(cache_file, 'rb') as handle:
            (outputs, targets, probs) = pickle.load(handle)
    else:
        dataset1 = ABSADataset(model1.tokenizer,
                              task_name=task,
                              data_name=data,
                              data_type=data_type,
                              args=args,
                              n_tuples=0,
                              wave=1,
                              max_len=args.max_seq_length)
        data_loader1 = DataLoader(dataset1,
                                 batch_size=args.eval_batch_size,
                                 num_workers=10)
        for i in range(0, 24):
            data_sample = dataset1[i]
            print(
                'Input1 :',
                model1.tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            '''print('Input :',
                  model.tokenizer.convert_ids_to_tokens(data_sample['source_ids']))'''
            print(
                'Output1:',
                model1.tokenizer.decode(data_sample['target_ids'],
                                 skip_special_tokens=True))
            print()
            
        device = torch.device('cuda:0')
        model1.model.to(device)
        model1.model.eval()

        for batch in data_loader1:
            # beam search
            outs = model1.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=1,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    model1.prefix_allowed_tokens_fn, task, data,
                    batch['source_ids']) if args.constrained_decode else None,
            )
            dec = [
                model1.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs.sequences
            ]
            target = [
                model1.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["target_ids"]
            ]
            orders.extend(dec)
            order_targets.extend(target)

        # first stage performance evaluation
        orders = [len(order.split(' [SSEP] ')) for order in orders]
        order_targets = [len(order.split(' [SSEP] ')) for order in order_targets]
        mse = mean_squared_error(orders, order_targets)   
        rmse = round(np.sqrt(mse), 2)
        accuracy = accuracy_score(orders, order_targets)
        with open("my_order_score.txt", 'a') as file:
            file.write(f"first_stage_result: {task}, {data}, RMSE : " + str(rmse) + '\n')
            file.write(f"first_stage_result: {task}, {data}, Acc : " + str(accuracy) + '\n')
        
        dataset2 = ABSADataset(model2.tokenizer,
                              task_name=task,
                              data_name=data,
                              data_type=data_type,
                              args=args,
                              n_tuples=orders,
                              wave=2,
                              max_len=args.max_seq_length)
        data_loader2 = DataLoader(dataset2,
                                 batch_size=args.eval_batch_size,
                                 num_workers=10)
        
        for i in range(0, 24):
            data_sample = dataset2[i]
            print(
                'Input_evaluate: ',
                model2.tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            print()
        model2.model.to(device)
        model2.model.eval()

        for batch in data_loader2:
            # beam search
            outs = model2.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=1,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    model2.prefix_allowed_tokens_fn, task, data,
                    batch['source_ids']) if args.constrained_decode else None,
            )
            dec = [
                model2.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs.sequences
            ]
            target = [
                model2.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["target_ids"]
            ]
            outputs.extend(dec)
            targets.extend(target)
            
        # save outputs and targets
        with open(cache_file, 'wb') as handle:
            pickle.dump((outputs, targets, probs), handle)
    
    _outputs = outputs # backup
    outputs = [] # new outputs
    new_targets = [] # new targets
    for i in range(len(targets)):
        output_quads = []
        output_quads.extend(extract_spans_para(seq=_outputs[i], seq_type='pred'))
        # recover output
        output = []
        target = []
        for q in output_quads:
            ac, at, sp, ot = q

            if tasks[i] in ["asqp", "acos", "memd"]:
                output.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")

            else:
                raise NotImplementedError
        new_targets.append(targets[i])
        
        # if no output, use the first path
        output_str = " [SSEP] ".join(
            output)
        outputs.append(output_str)

            
    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    #print("pred labels count", labels_counts)
    scores, all_labels, all_preds = compute_scores(outputs,
                                                   new_targets,
                                                   verbose=True)
    print("targets: ", new_targets[:20])
    print("outputs: ", outputs[:20])
    
    #print("scores: ", scores)
    return scores


def train_function(args):

    # training process
    if args.do_train:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)

        # sanity check
        # show one sample to check the code and the expected output
        #print(f"Here is an example (from the dev set):")
        dataset = ABSADataset(tokenizer=tokenizer,
                              task_name=args.task,
                              data_name=args.dataset,
                              data_type='train',
                              n_tuples=0,
                              wave=1,
                              args=args,
                              max_len=args.max_seq_length)
        dataset2 = ABSADataset(tokenizer=tokenizer,
                              task_name=args.task,
                              data_name=args.dataset,
                              data_type='train',
                              n_tuples=0,
                              wave=2,
                              args=args,
                              max_len=args.max_seq_length)
        for i in range(0, 10):
            data_sample = dataset[i]
            print(
                'Input_train1: ',
                tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            print(
                'Target_train: ',
                tokenizer.decode(data_sample['target_ids'],
                                 skip_special_tokens=True))
            print()
            data_sample2 = dataset2[i]
            print(
                'Input_train2: ',
                tokenizer.decode(data_sample2['source_ids'],
                                 skip_special_tokens=True))
            print(
                'Target_train2: ',
                tokenizer.decode(data_sample2['target_ids'],
                                 skip_special_tokens=True))
            print()
        print("\n****** Conduct Training ******")

        # initialize the T5 model
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name)  
        model_1 = T5FineTuner_1(args, tfm_model, tokenizer)
        # load data
        train_loader_1 = model_1.train_dataloader()
        # config optimizer
        t_total = ((len(train_loader_1.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(args.num_train_epochs))

        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total
        }

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=False)

        early_stop_callback = EarlyStopping(monitor="val_f1",
                                            min_delta=0.00,
                                            patience=20,
                                            verbose=True,
                                            mode="max")
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # prepare for trainer
        train_params = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[
                checkpoint_callback, early_stop_callback,
                TQDMProgressBar(refresh_rate=10), lr_monitor
            ],
        )

        trainer = pl.Trainer(**train_params)
        
        trainer.fit(model_1)
        
        print("\n****** Phase 1 finished! ******")
                
        tfm_model2 = copy.deepcopy(tfm_model)
        
        model_2 = T5FineTuner_2(args, tfm_model2, tokenizer)
        
        train_loader_2 = model_2.train_dataloader()
        
        t_total2 = ((len(train_loader_2.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(40))
        
        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total2
        }

        checkpoint_callback2 = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=False)

        early_stop_callback2 = EarlyStopping(monitor="val_f1",
                                            min_delta=0.00,
                                            patience=20,
                                            verbose=True,
                                            mode="max")
        lr_monitor2 = LearningRateMonitor(logging_interval='step')

        # prepare for trainer
        train_params2 = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=40,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[
                checkpoint_callback, early_stop_callback,
                TQDMProgressBar(refresh_rate=10), lr_monitor
            ],
        )
        trainer2 = pl.Trainer(**train_params2)
        trainer2.fit(model_2)
        print("\n****** Phase 2 finished! ******")
        # save the final model
        model_1.model.save_pretrained(os.path.join(args.output_dir, "first"))
        model_2.model.save_pretrained(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "first"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print("Finish training and saving the model!")

    if args.do_inference:
            print("\n****** Conduct inference on trained checkpoint ******")

            # initialize the T5 model from previous checkpoint
            #print(f"Load trained model from {args.output_dir}")
            '''print(
            'Note that a pretrained model is required and `do_true` should be False'
            )'''
            model_path1 = os.path.join(args.output_dir, "first")
            model_path2 = os.path.join(args.output_dir, "final")
            # model_path = args.model_name_or_path  # for loading ckpt

            tokenizer = T5Tokenizer.from_pretrained(model_path2)
            tfm_model1 = MyT5ForConditionalGeneration.from_pretrained(model_path1)
            tfm_model2 = MyT5ForConditionalGeneration.from_pretrained(model_path2)
            model1 = T5FineTuner_1(args, tfm_model1, tokenizer)
            model2 = T5FineTuner_2(args, tfm_model2, tokenizer)

            if args.load_ckpt_name:
                ckpt_path = os.path.join(args.output_dir, args.load_ckpt_name)
                #print("Loading ckpt:", ckpt_path)
                checkpoint = torch.load(ckpt_path)
                model1.load_state_dict(checkpoint["state_dict"])

            log_file_path = os.path.join(args.output_dir, "result.txt")
            time0 = time.time()
            # compute the performance scores
            with open(log_file_path, "a+") as f:
                config_str = f"seed: {args.seed}, beam: {args.beam_size}, constrained: {args.constrained_decode}\n"
                #print(config_str)
                f.write(config_str)
                scores = evaluate(model1,
                                  model2,
                                  args.task,
                                  args.dataset,
                                  data_type=args.eval_data_split)
                

                exp_results = "{} precision: {:.2f} recall: {:.2f} F1 = {:.2f}".format(
                    args.eval_data_split, scores['precision'], scores['recall'], scores['f1'])
                print(exp_results)
                f.write(exp_results + "\n")
                f.flush()
            infer_time = time.time() - time0
        
            with open("inference_time.txt", 'a') as file:
                file.write(str(infer_time) + '\n')
    return scores['f1']


if __name__ == '__main__':
    args = init_args()
    set_seed(args.seed)
    train_function(args)