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
from median_utils import *
from tqdm import tqdm
from pathlib import Path
import json
import torch._dynamo
torch._dynamo.config.suppress_errors = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    all_input_lengths = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    scratchpad_mode = args.mode
    get_input_length = lambda n: n * 4 + 2  # n: number of numbers
    

    output_path = Path("median_output")
    if not output_path.exists():
        output_path.mkdir()

    save_path = output_path / f"{scratchpad_mode}.json"
    if save_path.exists():
        with open(save_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    for input_length in all_input_lengths[::-1]:

        if get_input_length(input_length) in [int(k.split("-")[0]) for k in results]:
            print("********", input_length, "exists, go to next one", "********")
            continue
        accs = []
        print("********", "running number of numbers", input_length, "********")
        for seed in range(3):
            set_seed(seed)  
            n_positions = 1024

            batch_size = 64
            max_steps = 50_000
            
            lr = 3e-4
            n_layer = 3
            n_head = 4
            d_model = 256

            print("hyper-parameters", n_layer, n_head, d_model, lr)
            per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size 

            tokenizer = customTokenizer([str(i) for i in range(10)] + [";"])

            test_dataset = EvalDataset(MedianWithCoTDataset(tokenizer, (input_length, input_length), None, scratchpad_mode), (input_length+10)*20)
            train_dataset = MedianWithCoTDataset(tokenizer, (input_length, input_length), get_test_hash(test_dataset), scratchpad_mode)

            scratchpad_length = len(test_dataset[0][0]) - get_input_length(input_length) - 5
            print("scratchpad length", scratchpad_length)
            for i in range(3):
                print("\ninput example:")
                print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[i][0])))
                print("label example:")
                print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[i][1])))


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
                    output_dir="temp",    
                    overwrite_output_dir=True,
                    per_device_train_batch_size=per_device_bz,
                    per_device_eval_batch_size=per_device_bz,
                    max_steps=max_steps,
                    evaluation_strategy="steps",
                    eval_steps=3_000,
                    save_strategy="no",
                    logging_strategy="steps",
                    logging_steps=3_000,
                    learning_rate=lr,
                    weight_decay=0.01,
                    optim='adamw_torch',
                    lr_scheduler_type='linear',
                    warmup_steps=0,
                    report_to="none",
                    torch_compile=True,
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

            trainer.train()

            # final test, we allow model to fail on scratchpad, as long as it predicts the final answer.
            model = trainer.model
            model.eval()
            dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=data_collator)
            num_show = 1
            with torch.no_grad():
                up_to_sep = 2 + input_length * 4
                total_num = 0
                correct_num = 0
                for batch_input in dataloader:
                    input_ids = batch_input["input_ids"][:, :up_to_sep].to(model.device)
                    generated = model.generate(input_ids=input_ids, num_beams=1, do_sample=False, max_length=n_positions, eos_token_id=tokenizer.eos_token_id)
                    if num_show > 0:
                        print("first")
                        print(" ".join(tokenizer.convert_ids_to_tokens(input_ids[0].tolist())))
                        print(" ".join(tokenizer.convert_ids_to_tokens(generated[0].tolist())))
                        print("second")
                        print(" ".join(tokenizer.convert_ids_to_tokens(input_ids[1].tolist())))
                        print(" ".join(tokenizer.convert_ids_to_tokens(generated[1].tolist())))
                        print()
                        num_show -= 1
                    
                    for i in range(generated.size(0)):
                        total_num += 1
                        eos_pos = (generated[i] == tokenizer.eos_token_id).float().argmax().item()
                        if eos_pos == 0:    # no eos
                            continue
                        pred = generated[i, eos_pos-4: eos_pos+1]
                        ans = batch_input["input_ids"][i, -5:].to(model.device)
                        correct_num += (pred == ans).all().float().item()
            acc = correct_num / total_num
            print("final accuracy:", acc)
            accs.append(acc)

        results[f"{get_input_length(input_length)}-{scratchpad_length}"] = accs
        with open(save_path, "w") as f:
            json.dump(results, f)
