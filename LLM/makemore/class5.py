import torch
import torch.nn.functional as F
import random

# =================================================
# 1. 数据准备（block_size = 8，比之前长）
# =================================================
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

block_size = 8

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
print(f"Xtr.shape is {Xtr.shape}")
print(f"Ytr.shape is {Ytr.shape}")

                        
# =================================================
# 2. 自定义层（模仿 PyTorch nn 风格）
# =================================================
g = torch.Generator().manual_seed(2147483647) 

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming init
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
        
class BatchNorm1d:
    # dim: the numer of Features
    # eps: 修正分母
    # momentum: 动量,新数据的权重
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim) # scale
        self.beta = torch.zeros(dim) # shift
        self.running_mean = torch.zeros(dim) # 滚动均值
        self.running_var = torch.ones(dim) # 滚动方差
            
    def __call__(self, x):
        if x.ndim == 2: # (B, C)
            dim = 0 # 沿着第0维，算均值
        elif x.ndim == 3: # (B, T, C)
            dim = (0, 1) # 沿着第0, 1维度混在一起算均值
        #这两种情况都是一个特征通道C一个均值和方差
        if self.training:
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        # 归一化
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # 反归一化, 拉伸和平移
        self.out = self.gamma * xhat + self.beta

        # 影子统计量的滑动更新
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
        return self.out                   
        
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
class Embedding:
    # num_embedding: 词表的大小, 之前的abcd词表的大小就是27，26个字母+'.'
    # embedding_dim: 词表的维度, 需要多少维度来表述一个词, 之前就是10维
    def __init__(self, num_embeddings, embedding_dim):
        """等价于 C[X] 的封装"""
        # 创建词典
        self.weight = torch.randn((num_embeddings,embedding_dim), generator=g)    
        
    def __call__(self, IX):
        # IX: index
        self.out = self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]
# # 测试 Linear
# lin = Linear(10, 20)
# x = torch.randn(5, 10)
# y = lin(x)
# print(f"Linear: in {x.shape} -> out {y.shape}")    # (5, 10) -> (5, 20)

# # 测试 BatchNorm1d 2D
# bn2 = BatchNorm1d(20)
# y2 = bn2(y)
# print(f"BN 2D:  in {y.shape} -> out {y2.shape}")    # (5, 20) -> (5, 20)

# # 测试 BatchNorm1d 3D
# x3 = torch.randn(5, 4, 20)
# bn3 = BatchNorm1d(20)
# y3 = bn3(x3)
# print(f"BN 3D:  in {x3.shape} -> out {y3.shape}")    # (5, 4, 20) -> (5, 4, 20)

# =================================================
# 3. 搭 WaveNet
# =================================================
class FlattenConsecutive:
    """每n个连续时间步的特征拼成一个长向量"""
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []
class Sequential:
    """把层列表依次套娃执行"""
    # 例如：layers = [Embedding(...), Linear(...), Tanh(...)]
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        # 循环每一个网络层并且把每层的参数都放进去
        return [p for layer in self.layers for p in layer.parameters()]    
# =================================================
# 4. 训练（mini-batch + 分段 lr）
# =================================================
# ---- 模型超参数 + 搭 WaveNet ----
n_embd = 24
n_hidden = 128

model = Sequential([
    Embedding(vocab_size, n_embd),
    
    FlattenConsecutive(2), Linear(n_embd  * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    
    Linear(n_hidden, vocab_size),
])

# Lesson 3 同款修复：最后层缩小让初始 logits ≈ 0
with torch.no_grad():
    model.layers[-1].weight *= 0.1

# 收集参数 + 打开 autograd
parameters = model.parameters()
print(f"参数总数: {sum(p.nelement() for p in parameters)}")
for p in parameters:
    p.requires_grad = True

# 训练超参
batch_size = 32

for step in range(20000):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    logits = model(Xb)                    # ← 这是上面第 4 处新代码，但调用模式没变
    loss = F.cross_entropy(logits, Yb)
    
    for p in parameters:
        p.grad = None
    loss.backward()
    
    lr = 0.1 if step < 15000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad
    
    if step % 2000 == 0:
        print(f"step {step:5d} | loss {loss.item():.4f}")


# =================================================
# 5. 评估
# =================================================
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False

@torch.no_grad()
def split_loss(split):
    X, Y = {'train': (Xtr, Ytr), 'dev': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
    logits = model(X)
    return F.cross_entropy(logits, Y).item()

print(f"train: {split_loss('train'):.4f}")
print(f"dev:   {split_loss('dev'):.4f}")