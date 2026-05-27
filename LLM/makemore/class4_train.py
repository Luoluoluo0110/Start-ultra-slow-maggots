import torch
import torch.nn.functional as F
import random

# ===== 数据准备 =====
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

block_size = 3
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

# ===== 参数初始化 =====
n_embd, n_hidden = 10, 64
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / (n_embd * block_size)**0.5
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1
parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(f"参数总数: {sum(p.nelement() for p in parameters)}")
# 注意：没有 requires_grad = True，因为完全不用 autograd

# ===== 训练循环（用手算梯度）=====
batch_size = 32
n = batch_size
max_steps = 20000

with torch.no_grad():
    for step in range(max_steps):
        # mini-batch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # ===== forward =====
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar  = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        # ===== 手算 backward =====
        # cross-entropy 融合（Ex 2）
        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        
        # Linear 2 反向
        dh  = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        
        # tanh 反向
        dhpreact = (1.0 - h**2) * dh
        
        # BN gain/bias
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        
        # BN 融合反向（Ex 3）
        dhprebn = bngain * bnvar_inv / n * (
            n * dhpreact
            - dhpreact.sum(0, keepdim=True)
            - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0, keepdim=True)
        )
        
        # Linear 1 反向
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)
        
        # embedding 反向
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for ki in range(Xb.shape[0]):
            for kj in range(Xb.shape[1]):
                dC[Xb[ki, kj]] += demb[ki, kj]
        
        # ===== 参数更新 =====
        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
        lr = 0.1 if step < 10000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data -= lr * grad
        
        if step % 2000 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")
