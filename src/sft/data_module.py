import os
import sys
import json
from typing import Dict, Optional, Sequence, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
import datasets
from PIL import Image
import transformers
import random
import glob
from utils import get_model_identifiers_from_yaml


def preprocess_v1(tokenizer, input_ids, conversation, roles, ignore_index=-100):
    conversation = tokenizer.decode(input_ids[0])
    target = input_ids.clone()
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    cur_len = 1
    target[:, :cur_len] = ignore_index
    instruction = conversation.split(roles[1])[0].strip(" ")
    instruction_len = len(tokenizer(instruction + roles[1])['input_ids']) - 2
    target[:, cur_len : cur_len + instruction_len] = ignore_index
    # target[target == -100] = 0
    return target

def pad_sequence(sequences, padding_side='right', padding_value=0, max_len=None):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def pad_qformer_input_ids(input_ids_list, pad_token_id, max_length=50):
    padded_input_ids_list = []
    for input_ids in input_ids_list:
        if len(input_ids) > max_length:
            padded_input_ids = input_ids[:max_length]
        else:
            pad_tensor = [pad_token_id] * (max_length - len(input_ids))
            pad_tensor = torch.tensor(pad_tensor)
            padded_input_ids = torch.cat([input_ids, pad_tensor])
        padded_input_ids_list.append(padded_input_ids)
    
    padded_input_ids_list = [tensor.tolist() for tensor in padded_input_ids_list]
    padded_input_ids_tensor = torch.tensor(padded_input_ids_list)
    return padded_input_ids_tensor
    
class SafetyReasonDatasetQA(Dataset):
    def __init__(self, config, tokenizer, data_path=None, max_length=5120, split=None, processor=None, used_strategy=False, used_policy=False):
        super(SafetyReasonDatasetQA, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.used_strategy = used_strategy
        self.used_policy = used_policy
        
        self.data_path = data_path if data_path is not None else config.data_path
        try:
            with open(self.data_path, "r") as f:
                self.data = json.load(f)
        except:
            with open(self.data_path, "r") as f:
                self.data = [json.loads(line) for line in f.readlines()]
        
        # self.data = self.data[:200]
        self.model_configs = get_model_identifiers_from_yaml(config.model_family)
        
        self.samples = []
        for line in self.data:
            strategy, policy = None, None
            if len(line['templated_prompt']) > 0:
                question = line['templated_prompt']
                if self.used_strategy:
                    strategy = line['spec_strategy']
            else:
                question = line['ori_prompt']
                
            if self.used_policy and len(line['spec_safety_policy']):
                policy = line['spec_safety_policy']
                

            if strategy is not None and policy is not None:
                system_prompt = f"""You are a security orchestrator AI with the capabilities: Jailbreak strategy identification and Dynamic policy enforcement.  
I will provide you with a list of jailbreak strategies and customized safety policy. Your task is to retrieve the most relevant strategy from this list based on the user's instruction, analyze the user's intent, and extract the core user request.  
Next, you must analyze the extracted user request and strictly follow the provided safety policy to perform reasoning and generate a response.

Jailbreak Strategies:
{strategy}

Safety Policy:
{policy}"""
                
            elif strategy is not None and policy is None:
                system_prompt = f"""You are a security orchestrator AI with the capabilities: Jailbreak strategy identification.  
I will provide you with a list of jailbreak strategies. Your task is to retrieve the most relevant strategy from this list based on the user's instruction, analyze the user's intent, and extract the core user request.  
Next, you must analyze the extracted user request to perform reasoning and generate a response.

Jailbreak Strategies:
{strategy}"""
                
            elif strategy is None and policy is not None:
                system_prompt = f"""You are a security orchestrator AI with the capabilities: Dynamic policy enforcement.  
I will provide you with the customized safety policy. Your task is to analyze the user request and strictly follow the provided safety policy to perform reasoning and generate a response.

Safety Policy:
{policy}"""
            
            else:
                system_prompt = f"""You are a security orchestrator AI.
Your task is to analyze the user request and perform reasoning by referencing the appropriate safety policy to support your thoughts. Finally, generate a response."""

            self.samples.append(
                {
                   
                    "system_prompt": system_prompt,
                    "question": question,
                    "answer": line['cot_response'],
                    "label": line['label']
                }
            )

        random.shuffle(self.samples)

        print(
            f"There are {len(self.samples)} QA pairs for fine-tuning or evaluation!"
        )
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        system_prompt = self.samples[idx]['system_prompt']
        question = self.samples[idx]['question']
        answer = self.samples[idx]['answer']
        label = self.samples[idx]['label']
       
        # if "qwen" in self.config.model_family.lower():
        sources = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            sources,
            tokenize=False,
            add_generation_prompt=False
        )
        inputs = self.tokenizer([input_text], max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
        labels = preprocess_v1(self.tokenizer, inputs['input_ids'], input_text, roles)
                
        return {
            "input_ids": inputs['input_ids'].squeeze(0), 
            "attention_mask": inputs['attention_mask'].squeeze(0), 
            "labels": labels.squeeze(0), 
            "category": [label],
        }

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks



@dataclass
class custom_data_collator_perturbed(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        max_input_ids_shape = [max(tensor.size(dim) for tensor in input_ids) for dim in range(len(input_ids[0].size()))]
        max_label_shape = [max(tensor.size(dim) for tensor in labels) for dim in range(len(labels[0].size()))]

        pad_input_ids_list, pad_label_list = [], [] 
        for tensor in input_ids:
            padding_width = max_input_ids_shape[1] - tensor.size(1)
            padded_tensor = F.pad(tensor, (0, padding_width), 'constant', self.tokenizer.pad_token_id)
            pad_input_ids_list.append(padded_tensor)

        for tensor in labels:
            padding_width = max_label_shape[1] - tensor.size(1)
            padded_tensor = F.pad(tensor, (0, padding_width), 'constant', -100)
            pad_label_list.append(padded_tensor)
        
        input_ids = torch.stack(pad_input_ids_list)
        labels = torch.stack(pad_label_list)
        
        input_ids = input_ids[:, :, :self.tokenizer.model_max_length]
        labels = labels[:, :, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        for key in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']:
            if key in instances[0]:
                values = [instance[key].squeeze(1) for instance in instances]
                if all(x is not None and x.shape == values[0].shape for x in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
                
                if key == 'pixel_values' and len(values[0].shape) > 4:
                    batch[key] = batch[key].squeeze(1).unsqueeze(0)
                batch[key] = batch[key].squeeze(1)
        
        if "cross_attention_mask" in instances[0]:
            cross_attention_mask_list = [instance["cross_attention_mask"] for instance in instances]
            cross_attention_mask = pad_sequence(
                    cross_attention_mask_list, padding_side='right', padding_value=0
                )
            
            batch['cross_attention_mask'] = cross_attention_mask
          
        if 'qformer_input_ids' in instances[0]:
            qformer_input_ids = [instance['qformer_input_ids'] for instance in instances]
            if all(x is not None and x.shape == qformer_input_ids[0].shape for x in qformer_input_ids):
                batch['qformer_input_ids'] = torch.stack(qformer_input_ids)
            else:
                batch['qformer_input_ids'] = qformer_input_ids
                
            qformer_attention_mask = [instance['qformer_attention_mask'] for instance in instances]
            if all(x is not None and x.shape == qformer_attention_mask[0].shape for x in qformer_attention_mask):
                batch['qformer_attention_mask'] = torch.stack(qformer_attention_mask)
            else:
                batch['qformer_attention_mask'] = qformer_attention_mask
                
        return batch

@dataclass
class custom_data_collator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        for key in ['pixel_values', 'pixel_values_videos', 'aspect_ratio_ids', 'aspect_ratio_mask', 'image_grid_thw', 'video_grid_thw']:
            if key in instances[0]:
                values = [instance[key].squeeze(1) for instance in instances]
                if all(x is not None and x.shape == values[0].shape for x in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.cat(values)
                
                if 'pixel_values' in key and len(values[0].shape) > 4:
                    batch[key] = batch[key].squeeze(1).unsqueeze(0)
                else:
                    batch[key] = batch[key].squeeze(1)

        if "second_per_grid_ts" in instances[0]:
            # TODO: need to be fixed
            batch['second_per_grid_ts'] = [instance['second_per_grid_ts'][0][0] for instance in instances]

        if "cross_attention_mask" in instances[0]:
            cross_attention_mask_list = [instance["cross_attention_mask"][0] for instance in instances]
            cross_attention_mask = pad_sequence(
                    cross_attention_mask_list, padding_side='right', padding_value=0
                )
            
            batch['cross_attention_mask'] = cross_attention_mask
                
        if 'qformer_input_ids' in instances[0]:
            qformer_input_ids = [instance['qformer_input_ids'] for instance in instances]
            if all(x is not None and x.shape == qformer_input_ids[0].shape for x in qformer_input_ids):
                batch['qformer_input_ids'] = torch.stack(qformer_input_ids)
            else:
                batch['qformer_input_ids'] = qformer_input_ids
                
            qformer_attention_mask = [instance['qformer_attention_mask'] for instance in instances]
            if all(x is not None and x.shape == qformer_attention_mask[0].shape for x in qformer_attention_mask):
                batch['qformer_attention_mask'] = torch.stack(qformer_attention_mask)
            else:
                batch['qformer_attention_mask'] = qformer_attention_mask
        
        if 'category' in instances[0]:
            categories = [instance['category'][0] for instance in instances]
            batch['category'] = categories
        
        return batch

def pad_to_length(tensor, target_length, pad_value):
    padding_size = target_length - tensor.size(1)
    padding_tensor = torch.full((tensor.size(0), padding_size), pad_value)
    return torch.cat((tensor, padding_tensor), dim=1)

@dataclass
class custom_data_collator_forget(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        forget_instances, retain_instances = [instance[0] for instance in instances], [instance[1] for instance in instances]
        forget_input_ids, forget_labels = tuple([sample[key][0] for sample in forget_instances] for key in ("input_ids", "labels"))
        retain_input_ids, retain_labels = tuple([sample[key][0] for sample in retain_instances] for key in ("input_ids", "labels"))
        

        input_ids_max_length = -1
        for input_ids in forget_input_ids:
            input_ids_max_length = max(input_ids_max_length, input_ids.shape[-1])
        for input_ids in retain_input_ids:
            input_ids_max_length = max(input_ids_max_length, input_ids.shape[-1])
        
        labels_max_length = -1
        for labels in forget_labels:
            labels_max_length = max(labels_max_length, labels.shape[-1])
        for labels in retain_labels:
            labels_max_length = max(labels_max_length, labels.shape[-1])

        rets = []
        for data_type in ["forget", "retain"]:
            samples = forget_instances if data_type == "forget" else retain_instances
            input_ids, labels = tuple([sample[key][0] for sample in samples] for key in ("input_ids", "labels"))

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=-100)
            input_ids = pad_to_length(input_ids, input_ids_max_length, self.tokenizer.pad_token_id)
            labels = pad_to_length(labels, labels_max_length, -100)

            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
        
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

            if "cross_attention_mask" in samples[0]:
                cross_attention_mask_list = [instance["cross_attention_mask"][0] for instance in samples]
                cross_attention_mask = pad_sequence(
                        cross_attention_mask_list, padding_side='right', padding_value=0, max_len=input_ids.shape[-1]
                    )
                batch['cross_attention_mask'] = cross_attention_mask
                
            for key in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']:
                if key in samples[0]:
                    values = [instance[key].squeeze(1) for instance in samples]
                    if all(x is not None and x.shape == values[0].shape for x in values):
                        batch[key] = torch.stack(values)
                    else:
                        batch[key] = values
                    
                    
                    batch[key] = batch[key].squeeze(1)

                   

            rets.append(batch)
                
        return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(-1)
    return loss
