import torch
import numpy as np
import transformers
import datasets
import tiktoken
import wandb
from tqdm import tqdm

with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

#print(len(text))

chars = sorted(list(set(text)))

stoi = {ch:i for i,ch in enumerate(chars)}
iots = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([iots[i] for i in l])

#print(encode("hi there"))

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

n = int(0.8*len(data))
train_data = data[:n]
test_data = data[n:]