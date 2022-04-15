# -*- coding: utf-8 -*-
# from ctypes.wintypes import tagRECT
from sre_parse import Tokenizer
import torch
from torch.utils.data import Dataset
from itertools import chain
from transformers import DataCollatorForLanguageModeling
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS =  ["<s>", "</s>"]

class QEdataset(Dataset):
    def __init__(self, data, tokenizer, batch_first=True):
        self.data = data
        self.tokenizer = tokenizer

        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        # self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        src = self.data[index][0]
        tgt = self.data[index][1]  # tgt:[[],[]]
        tags = self.data[index][2]
        pe = self.data[index][3]
        da = self.data[index][4]
        # next_sentence_label = 0 if self.tokenizer.decode(label)=="0" else 1
        return self.process(src,tgt, tags,pe,da)
    
    def process(self, src, tgt, tags,pe, da):
        bos, eos = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS
        )
        instance = {}
        src = '</s> ' + src + ' </s>'
        tgt = '</s> ' + tgt + ' </s>'
        pe = '</s> ' + pe + ' </s>'
        input_pair = self.tokenizer(src,pe,add_special_tokens=False,return_tensors="pt") #0
        input_src = self.tokenizer(src,add_special_tokens=False,return_tensors="pt") #2
        # with self.tokenizer.as_target_tokenizer():
        input_target = self.tokenizer(tgt,add_special_tokens=False,return_tensors="pt")  #3
        # with self.tokenizer.as_target_tokenizer():
        input_pe = self.tokenizer(pe,add_special_tokens=False,return_tensors="pt") #4
        instance['input_pair'] = [input_pair["input_ids"][0],input_pair['attention_mask'][0]]
        instance['input_src'] = [input_src["input_ids"][0],input_src['attention_mask'][0]]
        instance['input_target'] = [input_target["input_ids"][0],input_target['attention_mask'][0]]
        instance['input_pe'] = [input_pe["input_ids"][0],input_pe['attention_mask'][0]]
        instance["labels"] = tags
        instance["da_labels"] = da
        return instance
    
    def collate(self, batch):
        # input_ids = [ instance["input_ids"] for instance in batch]
        # attention_mask = [instance["attention_mask"] for instance in batch ]
        # labels = [instance["labels"] for instance in batch ]
        # print(labels)
        # token_type_ids = pad_sequence(
        #     [
        #         torch.tensor(instance["token_type_ids"])
        #         for instance in batch
        #     ],
        #     batch_first=self.batch_first,
        #     padding_value=self.pad,
        # )
        input_pair = {}
        input_pair["input_ids"] = pad_sequence(
            [
                instance["input_pair"][0]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        input_pair["attention_mask"] = pad_sequence(
            [
                instance["input_pair"][1]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )
        input_src = {}
        input_src["input_ids"] = pad_sequence(
            [
                instance["input_src"][0]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        input_src["attention_mask"] = pad_sequence(
            [
                instance["input_src"][1]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )
        input_target = {}
        input_target["input_ids"] = pad_sequence(
            [
                instance["input_target"][0]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        input_target["attention_mask"] = pad_sequence(
            [
                instance["input_target"][1]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )
        input_pe = {}
        input_pe["input_ids"] = pad_sequence(
            [
                instance["input_pe"][0]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        input_pe["attention_mask"] = pad_sequence(
            [
                instance["input_pe"][1]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )

        labels = pad_sequence(
            [
                torch.tensor([float(instance["labels"])]) 
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        da_labels = pad_sequence(
            [
                torch.tensor([float(instance["da_labels"])]) 
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        return input_pair,labels,input_src,input_target,input_pe, da_labels
        
        # return instance
