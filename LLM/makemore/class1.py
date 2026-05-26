"""
==============================================================================
第 1 课 · Bigram 语言模型（计数版 + 神经网络版）
==============================================================================

目标：给定上一个字符，预测下一个字符。两条等价路线：
  A. 数计数 → 归一化 → 概率表 → 采样 / 算 loss      （没有学习）
  B. 神经网络 W(27,27) → softmax → 交叉熵 → 梯度下降  （有学习，loss 趋近 A）

最终 loss ≈ 2.45（信息论下限，bigram 这种"只看 1 个字符"的模型再聪明也突破不了）。

------------------------------------------------------------------------------
踩过的坑（复习时重点看）
------------------------------------------------------------------------------
1. VS Code Code Runner 插件吞输出 → 改用集成终端 (Ctrl+`)
2. 中文输入法的全角逗号 `，` ≠ 英文 `,`，Python 只认英文
3. 缩进错位 → `print` 写在 `for` 循环体里，跑了 32033 次
4. `torch(xs)` 错 → 应是 `torch.tensor(xs)`，torch 是模块不是函数
5. 用 `enumerate(stoi)` 当反向映射 → 错，应是 `stoi.items()`
6. multinomial 返回 tensor 忘了 `.item()` → `KeyError: tensor([10])`
7. PowerShell 里 `where python` 没输出 → where 是 Where-Object 的别名，要 `where.exe python`
8. 两个 Python 装一个，pip 装到 A，python 找 B → 用 `python -m pip install ...` 强制对齐

------------------------------------------------------------------------------
问过的好问题（揭示概念盲区）
------------------------------------------------------------------------------
Q1: `(N + 1).sum(...)` 分母为啥也加 1？
    A: 分子改了分母必须重算，否则每行加起来 ≠ 1 不是合法概率分布。

Q2: `torch.multinomial` 为啥返回索引而不是值？
    A: 索引代表"哪个字符"，值只是"概率本身"对你没用。

Q3: `W` 为啥是 (27, 27)？矩阵怎么乘的？
    A: 输入 27 维 (one-hot) → 输出 27 维 (logits)，所以 W = (in=27, out=27)。
       矩阵乘法形状规则：(A, B) @ (B, C) → (A, C)，中间维度消掉。

Q4: `counts = logits.exp()` 干嘛？
    A: logits 可以是任意实数（有负），`exp` 把所有数变正，再除以行和归一化 = softmax。

Q5: 为啥要前后加 `.`？
    A: 开头 `.` 让模型学"怎么开头"，结尾 `.` 让模型学"何时停"。
       block_size=1 加一个，block_size=3 开头要加 3 个（用 [0]*3 实现）。

Q6: W 为啥随机不能全零？
    A: 全零 → 所有梯度对称 → 27 行永远一样 → 学不到任何区分。Lesson 3 还会深挖。

Q7: 训练步数越多 loss 越小？
    A: 不会，会收敛到 ≈2.45 就停。这是 bigram 架构的信息天花板。

------------------------------------------------------------------------------
核心知识点（每一个后面都会再用）
------------------------------------------------------------------------------
- 字符 ↔ 整数：stoi / itos
- tensor 创建：torch.zeros / torch.randn / torch.tensor
- 切片 + 索引：N[i, j], N[ix1, ix2] += 1, fancy indexing P[arange, ys]
- 广播：N.float() / N.sum(1, keepdim=True)，keepdim 关键
- 概率与采样：torch.multinomial(p, 1, ..., generator=g).item()
- 损失：NLL = -log(probs[真实标签]).mean()
- One-hot：F.one_hot(xs, num_classes=27).float()
- 矩阵乘：xenc @ W（one-hot @ W 等价于 W[xs] 行查询）
- Softmax 三件套：exp → sum → 除
- 自动微分：requires_grad=True, loss.backward(), W.grad, W.data -= lr*W.grad
- 训练循环：forward → loss → backward → update（清梯度别忘）
- 随机可复现：torch.Generator().manual_seed(2147483647)
==============================================================================
"""

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1. 读数据 + 建字符映射
# ----------------------------------------------------------------------------
words = open('names.txt', 'r').read().splitlines()  # 32033 个名字

chars = sorted(list(set(''.join(words))))           # 26 个字母 a..z
stoi = {s: i + 1 for i, s in enumerate(chars)}      # 'a'->1, ..., 'z'->26
stoi['.'] = 0                                       # 特殊符号 . 占索引 0
itos = {i: s for s, i in stoi.items()}              # 反向映射 int -> str


# ----------------------------------------------------------------------------
# 2. 计数版：建 (27, 27) 计数表 N
# ----------------------------------------------------------------------------
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']                   # 头尾各加一个哨兵
    for ch1, ch2 in zip(chs, chs[1:]):              # 取所有相邻字符对
        N[stoi[ch1], stoi[ch2]] += 1


# ----------------------------------------------------------------------------
# 3. 计数版的概率表（加 1 平滑避免 log(0) = -inf）
#    采样和算 loss 已验证 loss ≈ 2.4544，注释保留作为参照
# ----------------------------------------------------------------------------
# P = (N + 1).float() / (N + 1).sum(1, keepdim=True)
#
# g = torch.Generator().manual_seed(2147483647)
# for _ in range(10):
#     out, ix = [], 0
#     while True:
#         ix = torch.multinomial(P[ix], 1, generator=g).item()
#         if ix == 0:
#             break
#         out.append(itos[ix])
#     print(''.join(out))
#
# log_likelihood, n = 0.0, 0
# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
#         n += 1
# print(f"average NLL(loss) = {-log_likelihood / n}")       # ≈ 2.4544


# ----------------------------------------------------------------------------
# 4. 神经网络版：把所有 bigram 打包成 (xs, ys) 两个大 tensor
# ----------------------------------------------------------------------------
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)                               # (228146,) 输入索引
ys = torch.tensor(ys)                               # (228146,) 真实标签


# ----------------------------------------------------------------------------
# 5. 初始化 W (27, 27) —— 整个网络只有一个参数矩阵
#    requires_grad=True 是开启 autograd 的钥匙
# ----------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


# ----------------------------------------------------------------------------
# 6. 训练循环
#    每步：forward → loss → backward → update
# ----------------------------------------------------------------------------
for k in range(100):
    # forward: one-hot → logits → softmax → probs
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W                               # (228146, 27)
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(xs.nelement()), ys].log().mean()

    # backward
    W.grad = None                                   # 清零旧梯度，必须
    loss.backward()

    # update
    W.data -= 50 * W.grad                           # .data 绕开计算图

    if k % 10 == 0:
        print(f"step {k:3d}: loss {loss.item():.4f}")

print(f"final loss = {loss.item():.4f}")            # ≈ 2.47，接近 2.4544
