import torch
from torch.utils.data import Dataset, IterableDataset
from torch.nn import CrossEntropyLoss
from copy import deepcopy
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Optional, Tuple, Union, Callable
import easydict
import os


class customTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # self.vocab = {"0": 0, "1": 1}
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        # this func is not used, since the data generator does not generate str
        # string is tokenized by white space
        if type(strings) == str:
            strings = [strings]
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def convert_tokens_to_ids(self, tokens: list[str]):
        return list(map(lambda x: self.vocab[x], tokens))
    
    def __len__(self):
        return len(self.vocab)
    


class MedianWithCoTDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], test_hash: set, scratchpad_mode: str):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.lengths = [i for i in range(self.range_min, self.range_max+1) if i % 2 == 1]
        assert self.lengths
        self.scratchpad_mode = scratchpad_mode
        self.test_hash = test_hash

    def digitalize(self, numbers):
        output = []
        for n in numbers:
            output.extend( list(f"{n:03d};") )
        return output

    def __iter__(self):
        while True:
            length = random.choice(self.lengths)
            
            numbers = random.choices(range(1000), k=length)
            sorted_numbers = sorted(numbers)[:length//2+1]
            
            instance = [self.tokenizer.bos_token_id]
            instance.extend(self.tokenizer.convert_tokens_to_ids( self.digitalize(numbers) ))
            instance.append(self.tokenizer.sep_token_id)
            pad_length = len(instance)

            if self.scratchpad_mode == "full-scratchpad":
                instance.extend(self.tokenizer.convert_tokens_to_ids( self.digitalize(sorted_numbers) ))
            elif self.scratchpad_mode == "no-scratchpad":
                instance.extend(self.tokenizer.convert_tokens_to_ids( self.digitalize(sorted_numbers[-1:]) ))
            elif self.scratchpad_mode.startswith("every"):
                interval = int(self.scratchpad_mode[5:])
                temp = sorted_numbers[:-1:interval] + sorted_numbers[-1:]
                instance.extend(self.tokenizer.convert_tokens_to_ids( self.digitalize(temp)))
            elif self.scratchpad_mode == "first-digit":
                instance.extend(self.tokenizer.convert_tokens_to_ids( [f"{n:03d};"[0] for n in sorted_numbers] ))
                instance.append(self.tokenizer.vocab[";"])
                instance.extend(self.tokenizer.convert_tokens_to_ids( self.digitalize(sorted_numbers[-1:]) ))
            else:
                raise NotImplementedError
            
            instance.append(self.tokenizer.eos_token_id)

            if self.test_hash is not None:
                if hash(tuple(instance)) in self.test_hash:
                    continue

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:pad_length] = [self.tokenizer.pad_token_id,] * pad_length   # bos + .. + sep 

            yield instance, label


class EvalDataset(Dataset):
    def __init__(self, d: IterableDataset, num_data: int):
        super().__init__()
        self.data = []
        for i, item in enumerate(d):
            if i >= num_data:
                break
            self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class customCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        input_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)

        [item.extend([self.pad_id,] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)
        [item.extend([self.pad_id,] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -100
        
        batch = {"input_ids": input_ids, "labels": labels}
        return batch
    


def compute_metrics(eval_preds):
    # scratchpad is also considered
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)
    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_test_hash(dataset: EvalDataset):
    hashes = set()
    for i in range(len(dataset)):
        input_ids = tuple(dataset[i][0])
        hashes.add(hash(input_ids))
    return hashes
