from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification, GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random
from copy import deepcopy
import string
import argparse
import itertools
import os
import re
from language_modeling2_train_adapted_NoOffset_PosEnc_FixedLen_VarLen import *

__file__ = __file__.split("/")[-1]
loss = torch.nn.CrossEntropyLoss(reduction='none') # removed ignore_index=0, 

import random
MAX_LEN = random.choice([10, 20, 30, 40, 50, 60, 70]) #, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]) #random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
COT = random.choice([5,10, 20, 30, 40, 50, 100, 200, 300, 400, 500])
#assert COT <= MAX_LEN
#assert (COT == 0) or (MAX_LEN % COT == 0)
iterations = 0

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, unroll=False, num_items_in_batch=None):
        if return_outputs and not unroll:
            return torch.zeros(1), torch.zeros(5)
        global iterations

        iterations += 1
      #  print(model)
#        print(inputs["input_ids"])
        states = []
        if inputs["input_ids"].size()[1] != MAX_LEN+2:
            print("WARNING", inputs["input_ids"].size()[1], MAX_LEN, COT)
        state = inputs["input_ids"][:,1]
        for i in range(COT):
            states.append(1+0*state)
        for i in range(2,MAX_LEN):
            state = torch.logical_xor(state, inputs["input_ids"][:,i]).detach()
        states.append(3+0*state)
        states.append(state)
        #        print(i, state)

        states = torch.stack(states, dim=1)
        states[:, :-2] = 0*states[:, :-2]
        inputs["input_ids_aug"] = torch.cat([inputs["input_ids"], states], dim=1)
        inputs["position_ids_aug"] = torch.cat([inputs["position_ids"], (inputs["input_ids"][:,-1:] + (torch.zeros(inputs["input_ids"].size()[0], COT).long() + 1 + torch.LongTensor(list(range(COT))).view(1,-1)).cuda()).long()], dim=1)
        VOCAB_SIZE=8
        COT_ = 3 #states.size()[1]
        #print(states.size(), COT, inputs["input_ids"].size(), inputs["input_ids_aug"].size())
#        assert (inputs["input_ids_aug"][:,-COT_-1] == 5).all(), (inputs["input_ids_aug"][0], )
 
        if not unroll:
            output = model(inputs["input_ids_aug"]).logits #, position_ids=positions.t()).logits.transpose(0,1).contiguous()
    
            nextTokenLogits = output[:,-COT_-1:-1].contiguous()
            targets = inputs["input_ids_aug"][:,-COT_:].contiguous()
            #print(nextTokenLogits[:,:,1])
            #print(targets)
            l_Last = loss(nextTokenLogits.view(-1,VOCAB_SIZE), targets.view(-1))
            accuracy = (torch.max(nextTokenLogits.view(-1,VOCAB_SIZE), dim=1).indices == targets.view(-1)).contiguous().view(-1, COT_)
            global accuracyAveraged
            accuracyAveraged = accuracy.float().mean().item()
            accuracy = accuracy.all(dim=1).float().mean().item()
            lossMean = l_Last.mean()
            if iterations % 1000 == 0:
              print(targets.size())
              print(torch.max(nextTokenLogits.view(-1,VOCAB_SIZE), dim=1).indices.view(targets.size()))
              print(targets)
              print(["Loss_Last", l_Last.mean().item(), "ACC", accuracy, accuracyAveraged, "COT", COT, "MAX_LEN", MAX_LEN])
              print("")
            #quit()
            if iterations == 1 or (((iterations + 10) % 1000) == 0) or random.random() < 0.001:
               accuracies.append(accuracy)
            return (lossMean, torch.max(nextTokenLogits.view(-1,8), dim=1).indices) if return_outputs else lossMean
        else:
            # 1) Pass everything except the last COT tokens
            output = model(inputs["input_ids_aug"][:, :-COT], use_cache=True)
            logits, kv_cache = output.logits, output.past_key_values
            
            # 2) Now generate the next COT tokens auto-regressively
            batch_size = inputs["input_ids_aug"].size(0)
            
            # We'll store the predicted tokens in nextInput (shape [batch_size, 1])
            nextInput = torch.argmax(logits[:, -1:, :], dim=-1)  # last logit along seq dim, pick best
            
            for j in range(COT-1):
                output = model(nextInput, past_key_values=kv_cache, use_cache=True)
                logits, kv_cache = output.logits, output.past_key_values
                # again, pick the best token from the last step
                nextInput = torch.argmax(logits[:, -1:, :], dim=-1)
            
            # 3) Compare final predicted token with ground truth
            accuracyOnFinal = (nextInput.squeeze(-1) == inputs["input_ids_aug"][:, -1]).float().mean().item()
            l_Last = loss(logits[:, -1].view(-1,VOCAB_SIZE), inputs["input_ids_aug"][:, -1].view(-1)).mean().item()
            print("Accuracy on final token:", accuracyOnFinal, l_Last)
            return accuracyOnFinal, l_Last

#        quit()
 #       labels = inputs.pop("labels")
  #      outputs = model(**inputs)
   #     logits = outputs.logits
    #    loss = nll_loss(logits, labels)

     #   return (loss, outputs) if return_outputs else loss

class myCallback2(TrainerCallback):
    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(test_length_ranges):
            if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) or (self.current_epoch == 1.0):  
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0: 
                    control.should_training_stop = True
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] >= threshold:
                    msg = "** " + msg
                    for key, value in self.latest_acc.items():
                        results[key].append(value)
                print(f"{n_layer}l{n_head}h{d_model}d\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), f"\t\tlr: {lr}", file=summary_f)
                summary_f.flush()


accuracies = []

#with open(f"OUTPUT/TEST.tsv", "a") as outFile:
#    print(COT, sum(accuracies[-20:])/20)

global accuracyAveraged
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, default=5)
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--regularize", type=float, default=0.0)
    parser.add_argument("--tasks", nargs='+', required=True)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(f"OUTPUT/{__file__}.tsv", "r") as inFile:
        done = [x.split("\t") for x in inFile.read().strip().split("\n")]
        done = [[x.strip() for x in y] for y in done]
        print(done)
        done = [x for x in done if x[0] == str(MAX_LEN) and x[1] == str(COT)]
        print(MAX_LEN, COT, done)
        assert len(done) < 2

    if not args.nope:
        task_arch = {"bin_majority": "1l1h16d",
                    "majority": "1l2h256d",
                    "bin_majority_interleave_sep": "2l4h256dsmalllr",
                    "unique_copy": "2l1h64d", 
                    "repeat_copy": "4l4h256d", 
                    "sort": "1l2h256dsmalllr", 
                    "parity": "2l2h256dsmalllr", 
                    "addition": "big", 
        }
    else:
        assert False
        task_arch = {"bin_majority": "1l1h16d",
                    "majority": "1l1h64d",
                    "bin_majority_interleave_sep": "big",
                    "unique_copy": "4l4h256d", 
                    "repeat_copy": "4l4h256d", 
                    "sort": "1l1h256d", 
                    "parity": "bigsmalllr", 
                    "addition": "big", 
        }

# CUDA_VISIBLE_DEVICES=0 python run_multiple_seeds2.py --tasks bin_majority majority bin_majority_interleave_sep
# CUDA_VISIBLE_DEVICES=1 python run_multiple_seeds2.py --tasks unique_copy
# CUDA_VISIBLE_DEVICES=2 python run_multiple_seeds2.py --tasks repeat_copy sort
# CUDA_VISIBLE_DEVICES=3 python run_multiple_seeds2.py --tasks parity
# regularize 0.0001

    train_length_range = (0, MAX_LEN)
    test_length_ranges = [train_length_range] + [(51, 100), (101, 150)]
    max_test_length = MAX_LEN+COT+100 # This needs to have the length of the CoT added on top
    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size 
    test_num = 500

    save_path = f"./lm-out-new-multi-run"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.nope:
        suffix = "-nope"
    elif args.regularize != 0:
        suffix = f"-reg{args.regularize}"
    else:
        suffix = ""
    

    for task in args.tasks:
        arch = task_arch[task]
        summary_f = open(os.path.join(save_path, f"{task}-average{suffix}.txt"), "w")
        print("\n\ntask: ", task, "\t", arch, "\n", file=summary_f)

        if not arch.startswith("big"):
            lr = 1e-3 if "smalllr" not in arch else 1e-4
            n_layer = int(re.search(r"(\d+)l", arch).group(1))
            n_head = int(re.search(r"l(\d+)h", arch).group(1))
            d_model = int(re.search(r"h(\d+)d", arch).group(1))
            max_steps = 30_000
            warmup_steps = 0
            threshold = 1.0
        else:
            lr = 1e-4 if "smalllr" not in arch else 3e-5
            n_layer = 12
            n_head = 12
            d_model = 768
            max_steps = 60_000
            warmup_steps = 3000
            threshold = 0.0



        print("hyper-parameters", n_layer, n_head, d_model, lr)

        results = {f"eval_len{test_range[0]}-{test_range[1]}_acc": [] for test_range in test_length_ranges}
        
        for seed in [random.randint(1,1000)]:
            torch.manual_seed(seed)
            random.seed(seed)



            if task == "parity":
                tokenizer = customTokenizer(["0", "1", "e", "o"])       # even, odd
                train_dataset = ParityDataset(tokenizer, train_length_range, max_test_length)

                test_dataset = EvalDataset(ParityDataset(tokenizer, train_length_range, max_test_length), test_num)

                n_positions = MAX_LEN + COT + 100  # bos, sep, ans, eos

            else:
                assert False
    
#            for j in range(3):
#                print("\ninput example:")
#                print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][j][0])))
#                print("label example:")
#                print(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][j][2])

        

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

            if args.nope:
                model = NoPEGPT2LMHeadModel(cfg)
            elif args.regularize != 0:
                assert False
            else:
                model = GPT2LMHeadModel(cfg)

            training_args = TrainingArguments(
                output_dir=os.path.join(save_path, "temp"),    
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
                warmup_steps=warmup_steps,
                report_to="none",
            )

            data_collator = customCollator(tokenizer.pad_token_id)


            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[myCallback2],
            )

            trainer.train()
        
            if len(results[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"]) == 1 : #args.num_run:
                break


        overall_norm = 0
        for name, param in model.named_parameters():
            if 'wpe' not in name:  # 'wpe' is often used for positional embeddings in GPT-2
                fro_norm = param.data.norm(p='fro')
                overall_norm += math.pow(fro_norm.item(), 2)
                print(f"Parameter: {name}, Frobenius Norm: {fro_norm.item()}")
            else:
                print(f"Skipping {name}")

        sharpness = {2e-3 : None, 5e-3 : None, 1e-5 : None, 1e-4 : None, 1e-2 : None, 1e-3 : None, 0.0 : None, 0.02 : None}
        sharpness_CE = {2e-3 : None, 5e-3 : None, 1e-5 : None, 1e-4 : None, 1e-2 : None, 1e-3 : None, 0.0 : None, 0.02 : None}
        import copy
        # 3. Create a clone of the model using copy.deepcopy.
        for noise_scale in sharpness:
          accuracy_perturbed = [0,0]
          CE_perturbed = [0,0]
          
           # 4. Perturb every parameter in the cloned model with a small Gaussian noise.
          for run in range(10):
           model_clone = copy.deepcopy(model)

           for param in model_clone.parameters():
             if 'wpe' not in name:  # 'wpe' is often used for positional embeddings in GPT-2
               param.data += noise_scale * torch.randn_like(param)
   
   
   
           trainer = CustomTrainer(
               model=model_clone,
               args=training_args,
               data_collator=data_collator,
               train_dataset=train_dataset,
               eval_dataset=test_dataset,
           )
           eval_dataloader = trainer.get_eval_dataloader(test_dataset)
   
           with torch.no_grad():
               for batch in eval_dataloader:
                   batch = {k: v.to(device) for k, v in batch.items()}
   #                print(batch)
               #    print(batch["input_ids"].size())
                   result_perturbed, ce_perturbed = (trainer.compute_loss(model_clone, batch, return_outputs=True, unroll=True))
                   accuracy_perturbed[0] += result_perturbed
                   accuracy_perturbed[1] += 1
                   CE_perturbed[0] += ce_perturbed
                   CE_perturbed[1] += 1
                   print("CE", ce_perturbed)
                   print(sharpness, sharpness_CE)
   
          sharpness[noise_scale] = accuracy_perturbed[0] / accuracy_perturbed[1]
          sharpness_CE[noise_scale] = CE_perturbed[0] / CE_perturbed[1]
        summary_f.flush()
        print(sharpness, sharpness_CE)
        summary_f.close()
        with open(f"OUTPUT/{__file__}.tsv", "a") as outFile:
            print(MAX_LEN, "\t", COT, "\t", accuracies[-1], "\t", accuracies, "\t", overall_norm, "\t", sharpness, "\t", sharpness_CE, file=outFile)
