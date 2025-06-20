import random
import json
import numpy as np
from itertools import permutations
import torch
from torch.utils.data import Dataset
from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration
import random
from t5_score import MyT5ForConditionalGenerationScore
from const import *
import pickle
import ast

def get_element_tokens(task):
    dic = {
        "exclude_A":
            ["[C]", "[O]", "[S]"],
        "exclude_C":
            ["[A]", "[O]", "[S]"],
        "exclude_O":
            ["[A]", "[C]", "[S]"],
        "exclude_S":
            ["[A]", "[C]", "[O]"],
        "asqp":
            ["[A]", "[O]", "[C]", "[S]"],
        "acos":
            ["[A]", "[O]", "[C]", "[S]"],
        "memd":
            ["[A]", "[O]", "[C]", "[S]"],
    }
    return dic[task]

def get_orders(task, data, args, sents, labels, data_type, wave):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = MyT5ForConditionalGenerationScore.from_pretrained(
        args.model_name).to(device)
    # Sort the views (permuations) based on entropy scores
    optim_orders_all = choose_best_order_global(sents, labels, model, tokenizer, device, task)
    
    return optim_orders_all


def cal_entropy(inputs, preds, model_path, tokenizer, device=torch.device('cuda:0')):
    all_entropy = []
    model = MyT5ForConditionalGenerationScore.from_pretrained(model_path).to(
        device)
    batch_size = 8
    _inputs = [' '.join(s) for s in inputs]
    _preds = [' '.join(s) for s in preds]
    for id in range(0, len(inputs), batch_size):
        in_batch = _inputs[id: min(id + batch_size, len(inputs))]
        pred_batch = _preds[id: min(id + batch_size, len(inputs))]
        assert len(in_batch) == len(pred_batch)
        tokenized_input = tokenizer.batch_encode_plus(in_batch,
                                                      max_length=200,
                                                      padding="max_length",
                                                      truncation=True,
                                                      return_tensors="pt")
        tokenized_target = tokenizer.batch_encode_plus(pred_batch,
                                                       max_length=200,
                                                       padding="max_length",
                                                       truncation=True,
                                                       return_tensors="pt")

        target_ids = tokenized_target["input_ids"].to(device)

        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        outputs = model(
            input_ids=tokenized_input["input_ids"].to(device),
            attention_mask=tokenized_input["attention_mask"].to(device),
            labels=target_ids,
            decoder_attention_mask=tokenized_target["attention_mask"].to(device))

        loss, entropy = outputs[0]
        all_entropy.extend(entropy)
    
    #print("cal_enrtopy: ", all_entropy)
    return all_entropy


def order_scores_function(quad_list, cur_sent, model, tokenizer, device, task):
    q = get_element_tokens(task)

    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)
    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_order = cur_order
            cur_target.append(each_q[cur_order.replace(" ", "  ") + " "][0])

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    tokenized_input = tokenizer.batch_encode_plus(all_inputs,
                                                  max_length=200,
                                                  padding="max_length",
                                                  truncation=True,
                                                  return_tensors="pt")
    tokenized_target = tokenizer.batch_encode_plus(all_targets,
                                                   max_length=200,
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

    target_ids = tokenized_target["input_ids"].to(device)

    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device))

    loss, entropy = outputs[0]
    results = {}
    for i, _ in enumerate(all_orders_list):
        cur_order = all_orders_list[i]
        results[cur_order] = {"loss": loss[i], "entropy": entropy[i]}
    return results


def choose_best_order_global(sents, labels, model, tokenizer, device, task):
    q = get_element_tokens(task)
    all_orders = permutations(q)
    all_orders_list = []
    return_orders = []
    scores = []

    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        scores.append(0)

    for i in range(len(sents)):
        label = labels[i]
        sent = sents[i]

        quad_list = []
        for _tuple in label:

            at, ac, sp, ot = get_task_tuple(_tuple, task)

            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            element_list = []
            for key in q:
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)

            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:4])
                    content.append(e[4:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        order_scores = order_scores_function(quad_list, sent, model, tokenizer,
                                             device, task)
        return_order = sorted(order_scores, key=lambda x: order_scores[x]["entropy"])
        return_orders.append(return_order)
        
    return return_orders

def read_line_examples_from_file(
                                 data_path,
                                 data_type,
                                 task_name,
                                 data_name,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels####orders
    Return List[List[word]], List[Tuple], List[string]
    """
    tasks, datas = [], []
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            tasks.append(task_name)
            datas.append(data_name)
            if line != '':
                words, tuples = line.split('####')

                words = words.split()
                sents.append(words)
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return tasks, datas, sents, labels


def get_task_tuple(_tuple, task):
    '''
    tuple -> (at, ac, sp, ot)
    '''
    if task == 'exclude_A':
        _, ac, sp, ot = _tuple
        at = None
    elif task == 'exclude_C':
        at, _, sp, ot = _tuple
        ac = None
    elif task == 'exclude_O':
        at, ac, sp, _ = _tuple
        ot = None
    elif task == 'exclude_S':
        at, ac, _, ot = _tuple
        sp = None
    elif task in ['asqp', 'acos', 'memd']:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError

    if sp:
        sp = sentword2opinion[sp.lower()] if sp in sentword2opinion \
            else senttag2opinion[sp.lower()]  # 'POS' -> 'good'
    if at and at.lower() == 'null':  # for implicit aspect term
        at = 'it'
    
    return at, ac, sp, ot


def add_prompt(sent, orders, task, data_name, args):
    sent_order = sent + orders
    return sent_order


def get_para_targets(sents, labels, data_name, data_type, task, args, n_tuples, wave):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []
    order_targets = []
    
    if data_type == 'test' and wave == 1:    # We don't have to calculate entropy score in Stage 1.
        with open('dummy.json', 'r') as file:
            orders = json.load(file)      
    else:
        orders = get_orders(task, data_name, args, sents, labels, data_type, wave)
        if data_type == 'train' and wave == 1:
            with open('dummy.json', 'w') as file:    # Store dummy views to reduce inference time in Stage 1.
                json.dump(orders, file)
 
    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]
        cur_sent_str = " ".join(cur_sent)

        # sort label by order of appearance
        # at, ac, sp, ot
        if args.sort_label and len(label) > 1:
            label_pos = {}
            for _tuple in label:
                at, ac, sp, ot = get_task_tuple(_tuple, task)

                # get last at / ot position
                at_pos = cur_sent_str.find(at) if at else -1
                ot_pos = cur_sent_str.find(ot) if ot else -1
                last_pos = max(at_pos, ot_pos)
                last_pos = 1e4 if last_pos < 0 else last_pos
                label_pos[tuple(_tuple)] = last_pos
            new_label = [
                list(k)
                for k, _ in sorted(label_pos.items(), key=lambda x: x[1])
            ]
            label = new_label

        quad_list = []
        for _tuple in label:
            at, ac, sp, ot = get_task_tuple(_tuple, task)
            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            token_end = 3

            element_list = []
            
            try:    
                for key in orders[i][0].split(" "):
                    element_list.append("{} {}".format(key, element_dict[key]))
            except:
                for key in orders[0][0].split(" "):
                    element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:token_end])
                    content.append(e[token_end:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)
        
        # Slice the orders based on the number of tuples (at inference time, the number of predicted views)
        order = orders[i][:n_tuples[i]] if n_tuples else orders[i][:min(len(label), 6)] # Limit the length of views
        order_prompt = ' [SSEP] '.join(order)
        order_targets.append(order_prompt)
        tar = []
        # One-to-one correspondence between views and tuples.
        for j in range(len(quad_list)):
            tar.append(quad_list[j][order[min(j, len(order) - 1)]][1])
        targets.append(" [SSEP] ".join(tar))
        new_sent = add_prompt(cur_sent, order_prompt.split(), task, data_name, args)
        new_sents.append(new_sent)
    
    print('new_sents: ', new_sents[:20])
    print('new_targets: ', targets[:20])
    print("get_para_targets - sents: ", len(new_sents))
    print("get_para_targets - targets: ", len(targets))
    return new_sents, targets, order_targets


def get_transformed_io(data_path, data_name, data_type, args, n_tuples, wave):
    """
    The main function to transform input & target according to the task
    """
    # training
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, data_type, args.task, args.dataset, args.lowercase)
    print(len(sents))

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]
    
    
    if wave == 1:
        new_inputs, targets, order_targets = get_para_targets(inputs, labels, data_name,
                                               data_type, args.first_stage_views,
                                               args, n_tuples, wave)
    else:
        new_inputs, targets, order_targets = get_para_targets(inputs, labels, data_name,
                                               data_type, args.task,
                                               args, n_tuples, wave)
    
    return inputs, new_inputs, targets, order_targets


class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 task_name,
                 data_name,
                 data_type,
                 args,
                 n_tuples,
                 wave,
                 max_len=256):
        self.data_path = f'{args.data_path}/{task_name}/{data_name}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.data_name = data_name
        self.data_type = data_type
        self.args = args
        self.wave = wave
        self.n_tuples = n_tuples
        
        
        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):
        ori_inputs, new_inputs, new_targets, order_targets = get_transformed_io(self.data_path,
                                                 self.data_name,
                                                 self.data_type,
                                                 self.args,
                                                 self.n_tuples,
                                                 self.wave)
        if self.wave == 1:
            inputs, targets = ori_inputs, order_targets
            target_max_length = 130
        else:
            inputs, targets = new_inputs, new_targets
            target_max_length = 1024 if self.data_type == "test" else self.max_len

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            try:
                target = targets[i]
            except:
                target = '[A] [C] [O] [S]'
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            
            # for ACOS Restaurant and Laptop dataset
            # the max target length is much longer than 200
            # we need to set a larger max length for inference

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=target_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)