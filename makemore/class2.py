"""
==============================================================================
第 2 课 · MLP 语言模型（Bengio 2003）
==============================================================================

目标：把 bigram"看 1 个字符"扩展到"看 block_size 个字符"，并用 embedding
代替 one-hot。loss 从 bigram 的 2.45 推进到 ≈ 2.1-2.3。

网络结构：
  3 个字符索引 (block_size=3)
      ↓  Embedding lookup C (27, 10)
  (B, 3, 10)
      ↓  .view(-1, 30)  拉平
  (B, 30)
      ↓  @ W1 (30, 200) + b1 → tanh
  (B, 200)
      ↓  @ W2 (200, 27) + b2
  (B, 27)  logits
      ↓  F.cross_entropy(logits, Yb)
  loss

总参数 = 27*10 + 30*200 + 200 + 200*27 + 27 = 11897

------------------------------------------------------------------------------
踩过的坑（复习时重点看）
------------------------------------------------------------------------------
1. itos 反向映射错写成 `{i: s for s, i in enumerate(stoi)}`
   → 这是按 dict 的 key 枚举，给的是 (下标, 键)，结果方向没翻。
   → 正确：`{i: s for s, i in stoi.items()}`
   → 规则：遍历 dict 几乎永远用 `.items()`，不要用 `enumerate(dict)`。

2. W2 漏了一个维度：`torch.randn(n_hidden, generator=g)`
   → 这是 1D shape (200,)，不是 2D 矩阵。
   → 正确：`torch.randn((n_hidden, 27), generator=g)`，注意元组括号。
   → 表现为参数总数偏少 5400、后面 logits 形状错。

3. seed 多打一位：`manual_seed(21474883647)`
   → 应是 `2147483647`（= 2^31 - 1）。多一位 8。不报错但跟视频对不上。

4. `F.cross_entropy(logits, Y.item())` ← `.item()` 位置错
   → `Y` 是多元素 tensor，`.item()` 只能用在单元素 tensor 上，会报错。
   → 正确：`F.cross_entropy(logits, Y).item()`，把 `.item()` 放到最外面，
     把返回的 loss tensor 转 Python 数字以便 f-string 格式化。

5. 之前残留的 demo 循环引用了 `X, Y`（已被 Xtr/Xdev/Xte 取代）
   → 旧代码引用已不存在的变量 → `NameError`。换 step 时及时清理 demo 代码。

------------------------------------------------------------------------------
问过的好问题（揭示概念盲区）
------------------------------------------------------------------------------
Q1: 为什么开头要 3 个 `.`？
    A: block_size = 3 → 模型需要 3 个字符的历史。开头不够时用 `.` 补齐
       (`context = [0] * block_size` 就是 3 个 `.`）。结尾 1 个 `.` 教会
       模型何时停。本质跟 bigram 头尾各 1 个 `.` 一样，只是 padding 长度
       随 block_size 变。

Q2: 为啥 random.seed 用 42？
    A: 文化梗，"生命宇宙终极答案" = 42。任意固定数都行，关键是固定。

Q3: build_dataset 函数里 `X.shape` 是哪个 X？
    A: 函数内部的局部变量 X。Python 函数有独立作用域，函数里的 X 和
       外面的 Xtr/Xdev/Xte 是两个世界，靠 return 沟通。

Q4: 拉平操作在哪？
    A: 藏在 `emb.view(-1, n_embd * block_size)` 一行里。可以拆成
       `emb_flat = emb.view(-1, 30); h = torch.tanh(emb_flat @ W1 + b1)`。

Q5: `ix` 是啥？
    A: 这一步 mini-batch 的 32 个样本编号。`Xtr[ix]` 按编号挑出对应行。

------------------------------------------------------------------------------
核心知识点（每一个 Transformer 都还会用）
------------------------------------------------------------------------------
- block_size：上下文窗口大小（决定模型能看多远的历史）
- Embedding lookup：C[X] 比 one-hot @ W 更紧凑、可学习 dense 表示
- 形状变换 .view(-1, d)：不复制数据，只重新解释内存
- Linear 层 = x @ W + b（带 bias，bigram 那节没用）
- 非线性激活 tanh：让多层 Linear 真的"多层"（否则等价于一层）
- F.cross_entropy：softmax + log + NLL 三合一，数值稳定
- Mini-batch：每步只采 32 个样本，速度飞起来；torch.randint
- Train / dev / test 三分（random.shuffle + 80/10/10）
- @torch.no_grad() 装饰器：评估时不建计算图，省内存
- 学习率分阶段：前期大步走 0.1，后期精调 0.01
- 多参数管理：parameters = [...]; 循环清梯度 / 循环更新

------------------------------------------------------------------------------
实测结果
------------------------------------------------------------------------------
- 初始 loss ≈ 25-27（随机权重太大，W ~ N(0,1) 输出极度尖锐）
- 1000 步后降到 ~3-4
- 10000 步 lr 衰减后稳定在 ~2.2-2.4
- 最终 train ≈ 2.32, dev ≈ 2.34（比 bigram 2.45 强）
- 采样名字明显比 bigram 像样：mora, kayah, kalin, riyah, ...
- 与 Karpathy 视频 ~2.10 的差距来自 Kaiming 初始化 + BatchNorm 还没做
  → Lesson 3 会修
==============================================================================
"""

import torch
import torch.nn.functional as F
import random


# ----------------------------------------------------------------------------
# 1. 读数据 + 建字符映射（跟 class1.py 一样）
# ----------------------------------------------------------------------------
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}              # 反向映射用 .items() 不是 enumerate
vocab_size = len(stoi)                              # 27


# ----------------------------------------------------------------------------
# 2. 上下文长度 + 数据集构造
# ----------------------------------------------------------------------------
block_size = 3

def build_dataset(words):
    """把每个名字滑窗成 (3 字符上下文, 下一个字符) 样本对"""
    X, Y = [], []
    for w in words:
        context = [0] * block_size                  # 开头 padding 用 . (索引 0)
        for ch in w + '.':                          # 名字末尾加结束符
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]            # 滑动窗口：去最旧、加最新
    X = torch.tensor(X)                             # (N, 3) 输入
    Y = torch.tensor(Y)                             # (N,) 标签
    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
    return X, Y


# ----------------------------------------------------------------------------
# 3. 训练/验证/测试三分（80/10/10），打乱后切片
# ----------------------------------------------------------------------------
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr,  Ytr  = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte,  Yte  = build_dataset(words[n2:])


# ----------------------------------------------------------------------------
# 4. 初始化所有参数
# ----------------------------------------------------------------------------
n_embd = 10                                         # 每个字符的 embedding 维度
n_hidden = 200                                      # 隐藏层宽度

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),              generator=g)  # 嵌入表
W1 = torch.randn((n_embd * block_size, n_hidden),   generator=g)  # 输入层 30->200
b1 = torch.randn(n_hidden,                          generator=g)
W2 = torch.randn((n_hidden, vocab_size),            generator=g)  # 输出层 200->27
b2 = torch.randn(vocab_size,                        generator=g)

parameters = [C, W1, b1, W2, b2]
print(f"参数总数: {sum(p.nelement() for p in parameters)}")        # 11897

for p in parameters:
    p.requires_grad = True


# ----------------------------------------------------------------------------
# 5. 训练循环（mini-batch + 分阶段学习率）
# ----------------------------------------------------------------------------
batch_size = 32

for step in range(20000):
    # mini-batch：每步随机抽 32 个样本
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward
    emb = C[Xb]                                      # (32, 3, 10)
    h = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)   # (32, 200)
    logits = h @ W2 + b2                             # (32, 27)
    loss = F.cross_entropy(logits, Yb)               # 自动 softmax + log + NLL

    # backward
    for p in parameters:
        p.grad = None
    loss.backward()

    # update（lr 分两段）
    lr = 0.1 if step < 10000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    if step % 1000 == 0:
        print(f"step {step:5d} / 20000 | loss {loss.item():.4f}")

print(f"final batch loss = {loss.item():.4f}")


# ----------------------------------------------------------------------------
# 6. 在完整 train / dev 集上算"真实" loss
# ----------------------------------------------------------------------------
@torch.no_grad()                                     # 不建计算图、不存梯度
def evaluate(split):
    X, Y = {'train': (Xtr, Ytr), 'dev': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
    emb = C[X]
    h = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()         # .item() 在最外面，返回 Python float

print(f"train loss: {evaluate('train'):.4f}")
print(f"dev   loss: {evaluate('dev'):.4f}")


# ----------------------------------------------------------------------------
# 7. 从训练好的 MLP 采样名字（自回归生成）
# ----------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out, context = [], [0] * block_size
    while True:
        emb = C[torch.tensor([context])]            # (1, 3, 10)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]                # 滑窗
        out.append(ix)
        if ix == 0:                                  # 遇到 . 停
            break
    print(''.join(itos[i] for i in out[:-1]))       # [:-1] 去掉末尾的 .
