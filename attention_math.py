import torch
import torch.nn as nn
from torch.nn import functional as F

# self-attention examples

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)

# version 1: averaging past context with for loops, the weakest form of aggregation
# we want x[b, t] = mean_{i<=t} x[b, i], so each token only talks to its past, never the future
xbow = torch.zeros((B, T, C)) # bow is short for "bag of words"
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t+1, C)
        xbow[b, t] = torch.mean(xprev, 0) # average over time -> (C,)

print(x[0])
print(xbow[0])

# version 2: the same weighted average as a batched matrix multiply
# row t of wei has 1/(t+1) in the first t+1 columns and 0 after, so wei @ x averages the past
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (T, T) @ (B, T, C) -> (B, T, C)
print(torch.allclose(xbow, xbow2))

# version 3: use softmax, the form self-attention uses
# future positions get -inf so softmax gives them 0 weight; the zeros elsewhere become a uniform average
# later these zeros are replaced by data-dependent affinities between tokens
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))

# version 4: self-attention, a single head
# every token emits a query (what am I looking for) and a key (what do I contain)
# affinities are dot products of queries with keys, so the weights now depend on the data
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # more channels so the projections have something to work with
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
# scale by 1/sqrt(head_size) so wei keeps unit variance and softmax doesn't collapse to one-hot
wei = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf')) # decoder block: no talking to the future
wei = F.softmax(wei, dim=-1)

v = value(x) # (B, T, 16), what each token communicates if attended to
out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
print(out.shape)
print(wei[0])
