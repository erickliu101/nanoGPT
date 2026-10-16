import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3 # self-attention can't tolerate very high learning rates
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)

with open("data/breakingbad/test.txt", 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
itos = {i:ch for i,ch in enumerate(chars)}
def decode(l):
    return ''.join([itos[i] for i in l])

# tokenized data from prepare.py
data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, 'bb_train.bin'), dtype=np.uint16).astype(np.int64))
test_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, 'bb_test.bin'), dtype=np.uint16).astype(np.int64))

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter, so register it as a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores ("affinities"), scaled so softmax doesn't collapse to one-hot
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # no talking to the future
        wei = F.softmax(wei, dim=-1)
        # weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token gets an embedding vector, and each position gets one too
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B, T)
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # attention has no notion of space, so add position information
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.sa_head(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens, since position embeddings only go that far
            idx_cond = idx[:, -block_size:]
            # gets predictions
            logits, loss = self(idx_cond)
            # focuses on last time step
            logits = logits[:, -1, :] # (B, C)
            # softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # one sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel().to(device)
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

# 1x1 tensor holding zero
idx = model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)
print(decode(idx[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # print(loss.item())
    print (iter, "/", max_iters)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))