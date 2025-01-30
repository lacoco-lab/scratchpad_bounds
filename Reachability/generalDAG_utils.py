import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.nn import CrossEntropyLoss
from copy import deepcopy
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config, TrainerCallback
from typing import Optional, Tuple, Union, Callable, List
from tqdm import tqdm, trange
import os
import wandb
import networkx as nx


class customTokenizer():
    def __init__(self, vocab: List[str]):
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

    def __call__(self, strings: Union[List[str], str], **kwargs):
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

    def convert_ids_to_tokens(self, ids: List[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def convert_tokens_to_ids(self, tokens: List[str]):
        return list(map(lambda x: self.vocab[x], tokens))
    
    def __len__(self):
        return len(self.vocab)
    

def random_dag(num_nodes: int, p: float):
    A = np.random.rand(num_nodes, num_nodes) < p
    A = np.triu(A, k=1)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    relabeling = np.random.permutation(num_nodes)
    return nx.relabel_nodes(G, {i: relabeling[i] for i in range(num_nodes)})


# returns two lists: (start_node, end_node, distance) and (start_node, end_node)
def dag_to_pairs(G: nx.DiGraph):
    paths = nx.all_pairs_shortest_path_length(G)
    connected, non_connected = [], []
    for start, others in paths:
        for node in range(G.number_of_nodes()):
            if node == start:
                continue
            elif node in others:
                connected.append((start, node, others[node]))
            else:
                non_connected.append((start, node))
    
    return connected, non_connected


# returns a graph, a list of connected pairs, a list of non-connected pairs, a list of probabilities for choosing connected pairs,
# and a hash of the graph
def sample_dag_example(num_nodes: int):
    while True:
        # if we use higher prob, graphs become too dense and the distances are too small
        prob = np.random.uniform(0.1, 0.6)
        dag = random_dag(num_nodes, prob)
        connected, non_connected = dag_to_pairs(dag)

        if len(connected) == 0 or len(non_connected) == 0:
            continue

        # stratification of distances
        distances = np.array([c[2] for c in connected])
        unique, counts = np.unique(distances, return_counts=True)
        mapping = dict(zip(unique, 1 / counts / len(unique)))
        p = np.apply_along_axis(lambda x: mapping[x[2]], 1, connected)

        return dag, connected, non_connected, p, nx.weisfeiler_lehman_graph_hash(dag)
    

def get_train_test_dags(num_nodes: int, num_train: int, num_test: int):
    test_samples = [sample_dag_example(num_nodes) for _ in range(num_test)]
    test_hashes = set([sample[-1] for sample in test_samples])

    train_samples = []
    while len(train_samples) < num_train:
        sample = sample_dag_example(num_nodes)
        if sample[-1] not in test_hashes:
            train_samples.append(sample)
        if len(train_samples) in (num_train * np.arange(0.1, 1.1, 0.1)).astype(int):
            print(len(train_samples))

    train_hashes = set([sample[-1] for sample in train_samples])
    print(f"Unique in test: {len(set(test_hashes))}, Unique in train: {len(set(train_hashes))}")
    
    return train_samples, test_samples
    

def edges_to_scratchpad(edges, start: int, end: int):
    scratchpad = [("N", start)]
    scratchpad_idx = 0
    visited = set()
    while scratchpad_idx < len(scratchpad):
        current_node = scratchpad[scratchpad_idx][1]
        if current_node != "N" and current_node not in visited:
            # list all edges starting in the current node
            scratchpad += [(l, r) for l, r in edges if l == current_node] + [(current_node, "N")]

            # without this the scratchpad would also work but it would slightly larger on average
            visited.add(current_node)

        # take the next edge in the scratchpad
        scratchpad_idx += 1
    
    # just searching if we've found the target node and trimming the scratchpad so that it's the end
    end_idx = [i for i, (l, r) in enumerate(scratchpad) if r == end]
    if end_idx:
        end_idx = end_idx[0]
        scratchpad = scratchpad[:end_idx+1]

    return scratchpad


class GeneralDAGWithCoTDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, dag_samples, scratchpad_mode, debug=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.dags, self.connected_pairs, self.non_connected_pairs, self.p, self.hashes = list(zip(*dag_samples))
        self.scratchpad_mode = scratchpad_mode
        self.debug = debug

    def convert_edges_to_tokens(self, edges: List[tuple]):
        tokens = []
        for s, t in edges:
            tokens.extend( [f"{s:02d}"] + [f"{t:02d}"] + [";"] )
        return tokens
    
    def get_example(self):
        dag_idx = np.random.choice(len(self.dags))
        dag = self.dags[dag_idx]
        if np.random.rand() < 0.5:
            label = "T"
            start, end, _ = self.connected_pairs[dag_idx][np.random.choice(len(self.connected_pairs[dag_idx]), p=self.p[dag_idx])]
        else:
            label = "F"
            start, end = self.non_connected_pairs[dag_idx][np.random.choice(len(self.non_connected_pairs[dag_idx]))]
        return dag, start, end, label
    
    def __iter__(self):
        while True:
            dag, query_i, query_j, label = self.get_example()

            edges = list(dag.edges)
            random.shuffle(edges)

            # yield edges, label, query_i, query_j
            # continue

            instance = [self.tokenizer.bos_token_id]
            # edge
            # instance.append(self.tokenizer.vocab["<edge>"])
            instance.extend(self.tokenizer.convert_tokens_to_ids( self.convert_edges_to_tokens(edges) ))
            # query
            instance.append(self.tokenizer.vocab["<query1>"])
            instance.extend(self.tokenizer.convert_tokens_to_ids( [f"{query_i:02d}"] ))
            instance.append(self.tokenizer.vocab["<query2>"])
            instance.extend(self.tokenizer.convert_tokens_to_ids( [f"{query_j:02d}"] ))

            instance.append(self.tokenizer.sep_token_id)
            pad_length = len(instance)

            # scratchpad
            if self.scratchpad_mode == "no-scratchpad":
                pass
            elif self.scratchpad_mode == "full-scratchpad":
                scratchpad = edges_to_scratchpad(edges, query_i, query_j)
                temp_tokens = []
                for (x, y) in scratchpad:
                    x = [f"{x:02d}"] if x != "N" else [x]
                    y = [f"{y:02d}"] if y != "N" else [y]
                    temp_tokens += [*x, *y, ";"]
                instance.extend(self.tokenizer.convert_tokens_to_ids( temp_tokens ))
            else:
                raise NotImplementedError
            
            ans_id = self.tokenizer.vocab[label]
            instance.append(ans_id)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:pad_length] = [self.tokenizer.pad_token_id,] * pad_length   # bos + .. + sep 

            if len(instance) >= 2048:
                print("warning: instance too long")
                continue

            if self.debug:
                yield instance, label, dag, query_i, query_j
                continue

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
        input_ids, labels = deepcopy(tuple(zip(*examples)))
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


def get_lengths(dataset: EvalDataset, tokenizer: customTokenizer):
    input_lengths = []
    scratchpad_lengths = []
    for i in range(len(dataset)):
        instance, label = dataset[i]
        input_length = label.count(tokenizer.pad_token_id)
        scratchpad_length = len(instance) - input_length - 2

        input_lengths.append(input_length)
        scratchpad_lengths.append(scratchpad_length)
    return torch.tensor(scratchpad_lengths, dtype=torch.float).mean().item(), max(scratchpad_lengths)


def get_test_hash(dataset: EvalDataset):
    hashes = set()
    for i in range(len(dataset)):
        input_ids = tuple(dataset[i][0])
        hashes.add(hash(input_ids))
    return hashes


def get_final_acc(model, dataset: EvalDataset, tokenizer: customTokenizer):
    from tqdm import tqdm

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=customCollator(tokenizer.pad_token_id))
    correct_num = up_to_sep = total_num = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            sep_idx = torch.where(batch["input_ids"] == tokenizer.sep_token_id)[1][0].item()
            up_to_sep += sep_idx
            input_ids = batch["input_ids"][:, :sep_idx+1].to(model.device)
            generated = model.generate(input_ids=input_ids, num_beams=1, do_sample=False, max_length=2048, eos_token_id=tokenizer.eos_token_id)
            if i < 5:
                print("example")
                print(" ".join(tokenizer.convert_ids_to_tokens(input_ids[0].tolist())) + "    |    " + " ".join(tokenizer.convert_ids_to_tokens(batch["input_ids"][0, -2:].tolist())))
                print(" ".join(tokenizer.convert_ids_to_tokens(generated[0].tolist())))

            pred = generated[0, -2:]
            ans = batch["input_ids"][0, -2:].to(model.device)

            if i < 5:
                print(f" {pred}    |    {ans}", end="\n\n")

            correct_num += (pred == ans).all().float().item()
            total_num += 1
    
    acc = correct_num / total_num
    up_to_sep = up_to_sep / total_num
    print("final accuracy:", acc, "avg input size:", up_to_sep)
    return acc, up_to_sep


class FinalAccCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, eval_steps=10000, target_acc=0.995):
        super().__init__()
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.target_acc = target_acc

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            model.eval()

            acc, avg_input_size = get_final_acc(model, self.test_dataset, self.tokenizer)

            print(f"[Step {state.global_step}] Final Accuracy: {acc:.4f}, Avg Input Size: {avg_input_size:.2f}")
            wandb.log({"eval/final_acc": acc}, step=wandb.run.step)

            if acc >= self.target_acc:
                print(f"Reached target accuracy of {self.target_acc:.2f}. Stopping training early!")
                control.should_training_stop = True

            model.train()