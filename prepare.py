import os
import pickle
import requests
import numpy as np
import torch

with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
    data = f.read()

#print(len(text))

chars = sorted(list(set(data)))

stoi = {ch:i for i,ch in enumerate(chars)}
iots = {i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([iots[i] for i in l])

#data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

n = int(0.8*len(data))
train_data = data[:n]
test_data = data[n:]

train_ids = encode(train_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'data', 'train.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'data', 'test.bin'))

