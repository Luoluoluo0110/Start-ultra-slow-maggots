"""
==============================================================================
第 4 课 · 手撕反向传播（Backprop Ninja）
==============================================================================

目标：抛弃 loss.backward()，手算每一步反向梯度，跟 PyTorch autograd 对照。
通关条件：
  Ex 1 — 30 个手算梯度全部 approx: True
  Ex 2 — cross-entropy 融合反向（1 行替代 10 步）
  Ex 3 — BatchNorm 融合反向（1 行替代 7 步）
  Ex 4 — 用手算梯度训练真网络，loss 跟 autograd 等价收敛（在 class4_train.py）

------------------------------------------------------------------------------
踩过的坑（复习时重点看）
------------------------------------------------------------------------------
1. `set('', join(words))` ← 错。`join` 是字符串方法。
   → 正确：`set(''.join(words))`

2. `context = ch[1:] + [stoi[ch]]` ← 错。`ch` 是单字符，`ch[1:]` = 空串。
   → 正确：`context = context[1:] + [stoi[ch]]`（滑动窗口）

3. `bnagin` ← 拼错。应是 `bngain`（gain = 增益）。

4. `bnraw = bndiff * bnvar` ← 数学错。这是 BN 公式核心一步。
   → 正确：`bnraw = bndiff * bnvar_inv`（乘 1/std，不是 var）

5. `p.requires_gard = True` ← 拼错。PyTorch 不报错，但 autograd 静默失效。
   → 正确：`p.requires_grad = True`（同款错也在 class3 出过）

6. `app = torch.allclose(dt, t.gard)` ← cmp() 函数里同款拼错。
   → 正确：`t.grad`

7. `bngain = torch.randn((1,n_hidden)) * 0.1 + 1.0` ← 漏 generator=g
   → 正确：`torch.randn((1,n_hidden), generator=g)`，可复现性

------------------------------------------------------------------------------
问过的好问题（揭示概念盲区）
------------------------------------------------------------------------------
Q1: `dlogprobs[range(n), Yb] = -1.0/n` 这一行没懂？
    A: loss = -挑出的 32 个数的均值。
       - 没被挑中的位置 → 改它 loss 不变 → 梯度 = 0
       - 被挑中的位置 → 改 ε，loss 变 -ε/n → 梯度 = -1/n
       (-) 来自 loss 公式的负号；(1/n) 来自 .mean()

Q2: 为啥后面的梯度不需要 `torch.zeros_like()`？
    A: **逐元素操作**（log/exp/*/...）的反向公式自动覆盖全部位置：
       `dprobs = (1/probs) * dlogprobs` — dlogprobs 是 0 的地方自然乘出 0
       只有**挑位置**（fancy indexing）或**循环累加**才要先建零容器：
       dlogprobs 和 dC 是这种情况。

Q3: 单样本梯度 `∂L/∂z_i = p_i - y_i` 对吗？
    A: 对（这是 Ex 2 公式的核心）。
       批 mean 后是 `(p_i - y_i) / n`，写成矩阵：
       `dlogits = (probs - one_hot(Y)) / n`

Q4: Ex 2 是不是不算中间值，直接数学推导？
    A: 对。把 loss 当成 logits 的整体函数代数化简：
       loss = -logits[Yb] + log(sum exp(logits))
       对 logits[k] 求导：(probs[k] - 1{k==Yb})，再除 n。
       一行替代前面 10 步链式求导。

Q5: cmp() 三件套的含义？
    A: exact = bit-by-bit 完全相等（罕见，浮点会差）
       approx = torch.allclose 数学等价（**这是目标**）
       maxdiff = 最大元素差（应该 0 或 1e-9 级别）

------------------------------------------------------------------------------
核心知识点（5 种反向结构 + 3 个融合公式）
------------------------------------------------------------------------------
反向传播的 5 种基本结构：
  | 前向                          | 反向                                |
  | y = f(x) 逐元素                | dx = f'(x) * dy                     |
  | Y = X @ W 矩阵乘               | dX = dY @ W.T, dW = X.T @ dY        |
  | s = x.sum(dim)                | dx = ones_like(x) * ds（广播）       |
  | c = a + b 广播（小 b）          | db = dc.sum(被广播维)               |
  | y = x[idx] 挑选 / 循环累加      | dx = zeros_like(x); dx[idx] += dy   |

三大融合公式（背下来）：
  Cross-entropy 融合：
    dlogits = (softmax(logits) - one_hot(Y)) / n

  tanh 反向（用输出复用）：
    dx = (1 - y**2) * dy    其中 y = tanh(x)

  BatchNorm 融合：
    dx = γ/(σ*n) * (n*dy - dy.sum(0)
                     - n/(n-1) * x_hat * (dy * x_hat).sum(0))

关键 PyTorch 招式：
  - retain_grad() 让中间张量留住 .grad 用于对照
  - torch.allclose(a, b) 容忍浮点误差的相等判定
  - F.one_hot(idx, num_classes=K) 整数索引转 one-hot
  - logits.max(1, keepdim=True).values / .indices
  - with torch.no_grad(): 包住不进计算图的代码

------------------------------------------------------------------------------
实测结果
------------------------------------------------------------------------------
- forward loss ≈ 3.5571
- Ex 1: 30 行 cmp 全部 exact: True | maxdiff: 0.0
- Ex 2: logits (fast) approx: True | maxdiff: 5e-9
- Ex 3: hprebn (fast) approx: True | maxdiff: 几乎 0
- Ex 4: 手算梯度训练 20000 步，loss ≈ 2.0-2.3，跟 autograd 等价
==============================================================================
"""

import torch
import torch.nn.functional as F
import random

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
Xte, Yte = build_dataset(words[n2:])

n_embd, n_hidden = 10, 64
g = torch.Generator().manual_seed(2147483647)
 
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / (n_embd * block_size) **0.5
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

batch_size = 32
n = batch_size

ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# C (27, 10) Xb (32, 3) 在C中找Xb对应的, 比如Xb第一行是[0, 0, 1], 去C的第0，0，1行，将(3 ,10)搬过来，重复32次，最后得到的形状是(32, 3, 10)
emb = C[Xb] 
# 将第一个参数设置成batch_size, 第二个参数自然就是n_embd * block_size = 30, 正好跟W1相乘
embcat = emb.view(emb.shape[0], -1) # (32, 30), W1 (30, 64), 这个64就是隐藏层的层数

#Linear 1 (Pre-BatchNorm)
hprebn = embcat @ W1 + b1 # (32, 64)

# BatchNorm(不使用.mean 手写)
bnmeani = 1/n * hprebn.sum(0, keepdim=True) # (1, 64) 均值
#算方差
bndiff = hprebn - bnmeani # 一个(32, 64) - (1, 64) ???, 广播机制，这个(1, 64)自动复制32次，成(32, 64)
bndiff2 = bndiff ** 2 # (32, 64)
bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True) # 计算方差(1， 64)
bnvar_inv = (bnvar + 1e-5)**-0.5 # (1, 64) 1/标准差，加 ε 数值稳定，标准差的倒数
bnraw = bndiff * bnvar_inv # (32, 64)这个就是标准化的数据 (xi - x)/标准差

hpreact = bngain * bnraw + bnbias # (32, 64)缩放平移后的最终输出

# 非线性
h = torch.tanh(hpreact) #(32, 64)

# Linear 2
logits = h @ W2 + b2 #(32, 27)
# 第 0 行（27 个数字）：是针对 Xb[0]（当前批次随机抓到的第 1 个样本） 的预测打分。
# Xb[0] 是 [0, 0, 1] $\rightarrow$ 对应字符就是 .、.、a（也就是某个名字最开始的第 1 个字母是 a）。
# 也就是(..a)后面出现数字的概率

# Cross-entroy loss (手写交叉熵损失函数)
logit_maxes = logits.max(1, keepdim=True).values # 找出每行得分最高的数字 得到(32, 1)
norm_logits = logits - logit_maxes #(32, 27) 所有数减去最大的数，最大的数变成0，其他数变成负数
counts = norm_logits.exp() # 指数运算 (32, 27)
counts_sum = counts.sum(1, keepdim=True) # 每行求和，得到(32, 1)
counts_sum_inv = counts_sum ** -1 # (32, 1)
probs = counts * counts_sum_inv # softmax操作, 每个元素取指数, 除以所有元素指数之和 (32, 27)

# 以softmax函数后的值作为预测值，交叉熵损失函数算出来的loss就等于负的真实值和log预测值相乘求和
logprobs = probs.log()


loss = -logprobs[range(n), Yb].mean() # Yb(32,) 以Yb为标准答案,算loss

print(f"loss = {loss.item():.4f}")

for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
    t.retain_grad()
loss.backward()
print("PyTorch autograd 跑完，开始手算对照")

# cmp()辅助函数, 对比手算的梯度和autograd的梯度
# 传入三个函数: s:字符串，比如W1， dt手动实现的梯度张量, PyTorch的张量

def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item() # 讲dt和t.grad对比，相同位置返回True，不同返回False，得到一个新的bool张量
    app = torch.allclose(dt, t.grad) #允许极小的误差1e-5
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | approx: {str(app):5s} | maxdiff: {maxdiff}")
    
# ==================== 26 个手算梯度 ====================
# 跟 forward 反过来：从 loss 出发，逐步往输入方向推

# # 1. dlogprobs: loss = -logprobs[range(n), Yb].mean()
# dlogprobs = torch.zeros_like(logprobs)
# dlogprobs[range(n), Yb] = -1.0/n
# cmp('logprobs', dlogprobs, logprobs)

# # 2. dprobs: logprobs = probs.log()，d(log x)/dx = 1/x
# dprobs = (1.0 / probs) * dlogprobs
# cmp('probs', dprobs, probs)

# # 3. dcounts_sum_inv: probs = counts * counts_sum_inv (广播 (32,27) * (32,1))
# #    counts_sum_inv 被广播到 27 列 → 反向要 sum(1)
# dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
# cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

# # 4. dcounts (来自 probs 路径，先算一半)
# dcounts = counts_sum_inv * dprobs

# # 5. dcounts_sum: counts_sum_inv = counts_sum ** -1
# #    d(x^-1)/dx = -x^-2 = -1/x^2
# dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
# cmp('counts_sum', dcounts_sum, counts_sum)

# # 6. dcounts (加上来自 counts_sum 路径): counts_sum = counts.sum(1, keepdim=True)
# #    每个 counts[i,j] 贡献 1 到 counts_sum[i,0]，反向就是 ones * dcounts_sum，广播自动
# dcounts += torch.ones_like(counts) * dcounts_sum
# cmp('counts', dcounts, counts)

# # 7. dnorm_logits: counts = norm_logits.exp()，d(exp x)/dx = exp x = counts
# dnorm_logits = counts * dcounts
# cmp('norm_logits', dnorm_logits, norm_logits)

# # 8. dlogits (来自 norm_logits 路径): norm_logits = logits - logit_maxes
# #    d/d(logits) = 1
# dlogits = dnorm_logits.clone()

# # 9. dlogit_maxes: norm_logits = logits - logit_maxes (广播 (32,1))
# #    d/d(logit_maxes) = -1，且要 sum(1) 因为广播
# dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
# cmp('logit_maxes', dlogit_maxes, logit_maxes)

# # 10. dlogits (加上来自 logit_maxes 路径): logit_maxes = logits.max(1, keepdim=True).values
# #     只有 argmax 那一位贡献 1，其他贡献 0 → one-hot
# dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
# cmp('logits', dlogits, logits)

# ===== Ex 2: cross-entropy 融合反向（一行替代前 10 步）=====

# ∂L_i / ∂z_i = (p_i - y_i) / n
# dlogits = (probs - one_hot(Y)) / n 数学推导

dlogits = F.softmax(logits, 1)        # (32, 27) 重算概率 = probs
dlogits[range(n), Yb] -= 1             # 真实答案位置 -1（= one_hot 减法）
dlogits /= n                            # 取平均

cmp('logits (fast)', dlogits, logits)   # 跟前面手算的 dlogits 对比

# 11. dh: logits = h @ W2 + b2
#     d(loss)/d(h) = d(loss)/d(logits) @ W2.T
dh = dlogits @ W2.T
cmp('h', dh, h)

# 12. dW2: 同上反向
dW2 = h.T @ dlogits
cmp('W2', dW2, W2)

# 13. db2: logits = h @ W2 + b2 (b2 广播 (27,) → (32, 27))，反向 sum(0)
db2 = dlogits.sum(0)
cmp('b2', db2, b2)

# 14. dhpreact: h = tanh(hpreact)，d(tanh x)/dx = 1 - tanh(x)^2 = 1 - h^2
dhpreact = (1.0 - h**2) * dh
cmp('hpreact', dhpreact, hpreact)

# # 15. dbngain: hpreact = bngain * bnraw + bnbias (bngain (1,64) 广播)
# dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
# cmp('bngain', dbngain, bngain)

# # 16. dbnraw: 同上的另一边
# dbnraw = bngain * dhpreact
# cmp('bnraw', dbnraw, bnraw)

# # 17. dbnbias: hpreact = ... + bnbias (广播)，反向 sum(0)
# dbnbias = dhpreact.sum(0, keepdim=True)
# cmp('bnbias', dbnbias, bnbias)

# # 18. dbndiff (来自 bnraw 路径): bnraw = bndiff * bnvar_inv
# dbndiff = bnvar_inv * dbnraw

# # 19. dbnvar_inv: bnraw = bndiff * bnvar_inv (bnvar_inv (1,64) 广播)
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

# # 20. dbnvar: bnvar_inv = (bnvar + 1e-5)**-0.5
# #     d/d(bnvar) = -0.5 * (bnvar + 1e-5)^-1.5
# dbnvar = -0.5 * (bnvar + 1e-5)**-1.5 * dbnvar_inv
# cmp('bnvar', dbnvar, bnvar)

# # 21. dbndiff2: bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)
# #     每个 bndiff2[i,j] 贡献 1/(n-1)，且 dbnvar (1,64) 广播回 (32,64)
# dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar
# cmp('bndiff2', dbndiff2, bndiff2)

# # 22. dbndiff (加上来自 bndiff2 路径): bndiff2 = bndiff ** 2
# #     d/d(bndiff) = 2 * bndiff
# dbndiff += 2 * bndiff * dbndiff2
# cmp('bndiff', dbndiff, bndiff)

# # 23. dhprebn (来自 bndiff 路径): bndiff = hprebn - bnmeani
# dhprebn = dbndiff.clone()

# # 24. dbnmeani: bndiff = hprebn - bnmeani (bnmeani (1,64) 广播)，反向 -sum(0)
# dbnmeani = (-dbndiff).sum(0, keepdim=True)
# cmp('bnmeani', dbnmeani, bnmeani)

# # 25. dhprebn (加上来自 bnmeani 路径): bnmeani = 1/n * hprebn.sum(0, keepdim=True)
# #     每个 hprebn[i,j] 贡献 1/n，广播
# dhprebn += (1.0/n) * torch.ones_like(hprebn) * dbnmeani
# cmp('hprebn', dhprebn, hprebn)

# ===== Ex 3: BatchNorm 融合反向 =====
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact
    - dhpreact.sum(0, keepdim=True)
    - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0, keepdim=True)
)

cmp('hprebn (fast)', dhprebn, hprebn)

# 26. dembcat: hprebn = embcat @ W1 + b1
dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

# 27. dW1:
dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)

# 28. db1:
db1 = dhprebn.sum(0)
cmp('b1', db1, b1)

# 29. demb: embcat = emb.view(emb.shape[0], -1)  (拉平的反向 = reshape 回去)
demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

# 30. dC: emb = C[Xb]  (fancy indexing 的反向 = scatter add)
#     对每个 (i, j)，C 的第 Xb[i,j] 行收到 demb[i, j, :]
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[i, j]
        dC[ix] += demb[i, j]
cmp('C', dC, C)



