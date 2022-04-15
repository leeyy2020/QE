from cProfile import label
import os
import math
import torch
import numpy as np
import random
import time
from argparse import ArgumentParser
from pprint import pformat
from torch.nn.parallel import DataParallel
from qe_en_de.dataloader import build_dataloaders
from itertools import chain
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
import logging
from model_bart import MBartForSequenceClassification,MBartForMultitask
from transformers import  MBart50TokenizerFast,MBartTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)
def train():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="models/",
        help="Path or URL of the model",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="If False train from scratch", default=True
    )
    parser.add_argument(
        "--data_path", type=str, default="data/WMT_2021/en-de/", help="Path or url of the dataset. "
    )
    parser.add_argument(
        "--save_path", type=str, default="model/en_de_2021/pytorch_model.bin", help="Path or url of the dataset. "
    )
    parser.add_argument(
        "--dataset_cache",
        action="store_true",
        help="use dataset cache or not",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of subprocesses for data loading",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use DataParallel or not",
    )
    args = parser.parse_args()
    saved_path = args.save_path
    saved_dir = os.path.dirname(saved_path)
    if not os.path.exists(saved_path):
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        # os.mknod(args.save_path)
    model_path = os.path.abspath(args.model_checkpoint)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info("Arguments: %s", pformat(args))
    logger.info("Prepare tokenizer, models and optimizer.")
    # load tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    logger.info("Prepare datasets")
    train_loader, val_loader = build_dataloaders(args, tokenizer, logger)
#     model = MBartForSequenceClassification.from_pretrained('../autodl-tmp/model/en_de_after_pes', num_labels=1)
#     model = MBartForSequenceClassification.from_pretrained('../autodl-tmp/model/en_de_pes/', num_labels=1)
    model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", num_labels=1)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.lr,
                eps=1e-6,
            )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=50000
    )

    model.to("cuda")
    # start training
    model.train()

    iteration = 0  
    eval_best_loss = 0
    # eval_best_loss = 0
    last_improve = 0  # 上一次提升
    flag = False  # 如果flag，则表示很久没有提升，结束训练
    start_time = time.time()
    i = 0
    for epoch in range(args.n_epochs):
        print('Epoch: [{}/{}]'.format(epoch + 1, args.n_epochs))

        for i, batch in enumerate(train_loader):
            src_input_ids, src_attention_mask = batch[2]["input_ids"].to("cuda"),batch[2]["attention_mask"].to("cuda")
            targe_input_ids, target_attention_mask = batch[3]["input_ids"].to("cuda"), batch[3]["attention_mask"].to("cuda")
            labels = batch[1].to("cuda")
            labels_da = batch[5].to("cuda")
            loss = model(input_ids = src_input_ids,attention_mask=src_attention_mask,decoder_input_ids = targe_input_ids,decoder_attention_mask=target_attention_mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if iteration % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            # 每100个iteration计算eval loss
            if iteration % 200 == 0 :
                pearson_score, spearman_score, mae, rmse = evaluate(model, val_loader)
                eval_loss = pearson_score * spearman_score
                if eval_loss > eval_best_loss:
                    eval_best_loss = eval_loss
                    torch.save(model.state_dict(), args.save_path)
                    improve = '*'
                    last_improve = iteration
                else:
                    improve = ''
                end_time = time.time()
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Val Loss: {2:>5.3},pearson_score: {3:>5.3},spearman_score: {4:>5.3},mae: {5:>5.3},rmse: {6:>5.3}, Time: {7} {8}'
                print(msg.format(iteration, loss.item(), eval_loss,pearson_score,spearman_score,mae,rmse, end_time - start_time, improve))
                model.train()
            iteration += 1
            if iteration - last_improve > 5000:  # 大概1-2个epoch
                print('No improve for a long time, auto stopping...')
                flag = True
                break
        if flag:
            break

def evaluate(model, valid_loader):
        model.eval()
        with torch.no_grad():
            outputs = []
            y_true = []
            for batch in valid_loader:
                src_input_ids, src_attention_mask = batch[2]["input_ids"].to("cuda"),batch[2]["attention_mask"].to("cuda")
                targe_input_ids, target_attention_mask = batch[3]["input_ids"].to("cuda"), batch[3]["attention_mask"].to("cuda")
                labels = batch[1].to("cuda")
                pre = model(input_ids = src_input_ids,attention_mask=src_attention_mask,decoder_input_ids = targe_input_ids,decoder_attention_mask=target_attention_mask, labels=labels).logits
                # for i in range(len(output)):
                for p in pre:
                    outputs.extend(p.cpu().numpy())
                for label in labels:
                    y_true.extend(label.cpu().numpy())
            hter_scores = np.array(y_true)
            output_hters = np.array(outputs)

            pearson_score = pearsonr(hter_scores, output_hters)[0]
            spearman_score = spearmanr(hter_scores, output_hters)[0]
            mae = np.mean(abs(hter_scores - output_hters))
            rmse = math.sqrt(np.mean((hter_scores - output_hters)**2))
        # return total_loss / len(valid_loader)
        return pearson_score, spearman_score, mae, rmse# return f1_MULTI  use f1_multi to choose model

if __name__ == '__main__':
    train()