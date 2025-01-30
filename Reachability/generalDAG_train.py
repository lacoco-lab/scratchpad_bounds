from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification, GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
import random
from copy import deepcopy
import string
import argparse
import itertools
import os
import re
from generalDAG_utils import *
from tqdm import tqdm
from pathlib import Path
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--vertex-num", type=int)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    scratchpad_mode = args.mode
    
    results = {}

    output_path = Path("generalDAG_output")
    if not output_path.exists():
        output_path.mkdir()
    
    vertex_num = args.vertex_num
    assert vertex_num <= 100, "we are using two digits to represent a vertex"
    accs = []
    avg_input_sizes = []
    print("********", "running DAG with", vertex_num, "vertex ********")
    avg_scratchpad_length = []
    max_scratchpad_length = []
    for seed in range(3):
        set_seed(seed)
        n_positions = 2048 # not important, as long as it's big enough

        batch_size = 64
        max_steps = 50_000
        
        lr = 3e-4
        n_layer = 4
        n_head = 4
        d_model = 256

        print("hyper-parameters", n_layer, n_head, d_model, lr)

        tokenizer = customTokenizer([f"{i:02d}" for i in range(100)] + ["N", ";", "<query1>", "<query2>", "F", "T"]) # ["<edge>", "->", ";", "<query>", "F", "T"])

        n_test_dags = 100
        n_train_dags = 30000

        train_samples, test_samples = get_train_test_dags(vertex_num, n_train_dags, n_test_dags)

        train_dataset = GeneralDAGWithCoTDataset(tokenizer, train_samples, scratchpad_mode)
        test_dataset = EvalDataset(GeneralDAGWithCoTDataset(tokenizer, test_samples, scratchpad_mode), 500)
        for i in range(3):
            print("\ninput example:")
            print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[i][0])))
            print("label example:")
            print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[i][1])))

        avg_scr_len, max_scr_len = get_lengths(test_dataset, tokenizer)
        avg_scratchpad_length.append(avg_scr_len)
        max_scratchpad_length.append(max_scr_len)

        cfg = GPT2Config(vocab_size=len(tokenizer), 
                    n_positions=n_positions,
                    n_embd=d_model,
                    n_layer=n_layer,
                    n_head=n_head,
                    bos_token_id=tokenizer.bos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    attn_pdrop=0,
                    resid_pdrop=0,
                    embd_pdrop=0,
                    )

        model = GPT2LMHeadModel(cfg)
        training_args = TrainingArguments(
            output_dir=f"temp/{vertex_num}-{scratchpad_mode}-{seed}",    
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_steps=max_steps,
            evaluation_strategy="steps",
            eval_steps=3_000,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=100,
            learning_rate=lr,
            weight_decay=0.01,
            optim='adamw_torch',
            lr_scheduler_type='linear',
            warmup_steps=0,
            report_to="wandb",
        )

        data_collator = customCollator(tokenizer.pad_token_id)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        final_acc_callback = FinalAccCallback(test_dataset, tokenizer, eval_steps=9_000)
        trainer.add_callback(final_acc_callback)

        trainer.train()

        # final test, we allow model to fail on scratchpad, as long as it predicts the final answer.
        model = trainer.model
        model.eval()

        acc, avg_input_size = get_final_acc(model, test_dataset, tokenizer)
        accs.append(acc)
        avg_input_sizes.append(avg_input_size)

    avg_scratchpad_length = sum(avg_scratchpad_length) / len(avg_scratchpad_length)
    max_scratchpad_length = max(max_scratchpad_length)
    avg_input_size = sum(avg_input_sizes) / len(avg_input_sizes)

    results[f"{vertex_num}-{avg_input_size:.4f}-{avg_scratchpad_length:.4f}-{max_scratchpad_length}"] = accs
    print(results)
    with open(output_path / f"{vertex_num}-{scratchpad_mode}.json", "w") as f:
        json.dump(results, f)