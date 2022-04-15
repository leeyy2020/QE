# -*- coding: utf-8 -*-
import torch
from transformers import cached_path
import json
from .dataset import QEdataset
from torch.utils.data import DataLoader
import os
import random
def get_data_from_file(data_path, dtype):
    assert (dtype == 'train' or dtype == 'test')
    if dtype == 'train':
        src_file = os.path.join(data_path, 'train.src')
        tgt_file = os.path.join(data_path, 'train.mt') 
        tag_file = os.path.join(data_path, 'train.hter')
        pe_file = os.path.join(data_path, 'train.pe')
        da_file = os.path.join(data_path, 'train.da')
    elif dtype == 'test':
        src_file = os.path.join(data_path, 'test.src')
        tgt_file = os.path.join(data_path, 'test.mt')
        tag_file = os.path.join(data_path, 'test.hter')
        pe_file = os.path.join(data_path, 'test.mt')
        da_file = os.path.join(data_path, 'test.da')
    all_data = []
    with open(src_file, "r", encoding="utf-8") as f1, \
        open(tgt_file, 'r', encoding='utf-8') as f2, \
        open(tag_file, 'r', encoding='utf-8') as f3, \
        open(pe_file, 'r', encoding='utf-8') as f4, \
        open(da_file, 'r', encoding='utf-8') as f5:
        src_lines = []
        tgt_lines = []
        tags_lines = []
        pe_lines = []
        da_lines = []
        for line1,line2,line3,line4,line5 in zip(f1,f2,f3,f4,f5):
            src_lines.append(line1.strip())
            tgt_lines.append(line2.strip())
            tags_lines.append(line3.strip())
            pe_lines.append(line4.strip())
            da_lines.append(line5.strip())
        
        assert len(src_lines) == len(tgt_lines) == len(tags_lines) == len(pe_lines)
        print(len(src_lines))
        for i in range(len(src_lines)):
            item = []
            item.append(src_lines[i])
            item.append(tgt_lines[i])
            item.append(tags_lines[i])
            item.append(pe_lines[i])
            item.append(da_lines[i])
            all_data.append(item)
    return all_data

def get_data(tokenizer, dataset_path, dataset_cache, logger):
    train_cache_dir = "cache/dataset_cache_train_" + type(tokenizer).__name__
    dev_cache_dir = "cache/dataset_cache_dev_" + type(tokenizer).__name__
    if False:
        logger.info("Load tokenized dataset from cache at %s", train_cache_dir)
        train_dataset = torch.load(train_cache_dir)
        dev_dataset = torch.load(dev_cache_dir)
    else:
        logger.info("Download dataset from %s", dataset_path)
        # 训练数据获取
        train_data = get_data_from_file(os.path.join(dataset_path, 'train'), 'train')
        dev_data = get_data_from_file(os.path.join(dataset_path, 'test'), 'test')
        print(len(train_data))
        logger.info("Tokenize and encode the dataset")
        # def src_tokenize(obj):
        #     if isinstance(obj, str):
        #         return src_tokenizer.convert_tokens_to_ids(src_tokenizer.tokenize(obj))
        #     if isinstance(obj, float):
        #         return obj
        #     return list(src_tokenize(o) for o in obj)
        train_dataset = []
        for item in train_data:
            ids_item = []
            ids_item.append(item[0])  #src
            ids_item.append(item[1])  #tgt
            ids_item.append(item[2])  #hter
            ids_item.append(item[3])  #pe
            ids_item.append(item[4])
            train_dataset.append(ids_item)
        dev_dataset = []
        for item in dev_data:
            # print(item)
            ids_item = []
            ids_item.append(item[0])
            ids_item.append(item[1])  # [[...], [...]]
            ids_item.append(item[2])
            ids_item.append(item[3])
            ids_item.append(item[4])
            dev_dataset.append(ids_item)
        # train_dataset = tokenize(train_data)
        # torch.save(train_dataset, train_cache_dir)
        # torch.save(train_dataset, dev_cache_dir)
    return train_dataset, dev_dataset

def build_dataloaders(args, tokenizer, logger): 
    logger.info("Build train and validation dataloaders")
    train_dataset, dev_dataset = get_data(tokenizer, args.data_path, args.dataset_cache, logger)
    train_dataset, valid_dataset = (
        QEdataset(train_dataset, tokenizer),
        QEdataset(dev_dataset, tokenizer),
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=None,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=None,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        batch_size=args.valid_batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader
