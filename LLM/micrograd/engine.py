"""
手写 micrograd 引擎

============ 学习过程中踩过的坑（按犯错顺序）============

[Python 语法]
1. set(_children())  ❌   —— _children 是元组不是函数，写 set(_children) 即可
2. (self)            ❌   —— 加了括号但没逗号，不是元组只是 self 本身
   (self,)           ✓   —— 单元素元组必须带逗号
3. '**{other}'       ❌   —— 普通字符串里 {other} 就是字面字符，不会替换
   f'**{other}'      ✓   —— 加 f 前缀才是 f-string，{} 内会求值
4. assert other isinstance(...)  ❌   —— assert 后只能跟一个表达式
   assert isinstance(...)        ✓

[函数对象 vs 调用]
5. out._backward = _backward()   ❌   —— 立刻执行函数，挂上去的是返回值 None
   out._backward = _backward     ✓   —— 把函数本身挂上去，留到反向时再调
6. for child in v          ❌   —— Value 对象本身不可迭代
   for child in v._prev    ✓   —— 要遍历的是父节点集合

[反向传播逻辑]
7. 方向：out 是上游，反向时把 out.grad 累加到父节点的 grad，不是反过来
8. 用 +=  不是 =  —— 同一个 Value 可能被多处使用，梯度必须累加
9. 定义了 _backward 闭包必须绑定到 out._backward 上，不绑等于白写
10. _backward 闭包不需要 return，它的作用是修改 .grad 的副作用

[算子特定]
11. 一元算子（pow / relu）的 _children 只放 (self,)，不放常数 / 不存在的 other；
    放错了 step6 的拓扑遍历会把常数当 Value 处理，AttributeError
12. pow 反向公式：k * a.data**(k-1) * out.grad —— 不要复制粘贴加法的公式
13. __sub__ 里 self 是减号左边的，写 self + (-other)
    __rsub__ 里 self 是减号右边的，写 other + (-self) —— 这是最容易写反的死区
14. __radd__ / __rmul__ 写 self + other / self * other 而不是 other + self；
    后者会触发死循环：数字 + Value 又会回到 __radd__

============ 核心设计思想 ============

- 每个 Value 节点在被创造时，就把"我以后该怎么反向"打包成闭包挂在 _backward 上
- backward() 只负责调度：拓扑序 → 火种 grad=1 → 逆序调用每个节点的 _backward
- 算子之间通过 _prev 形成 DAG，反向时挨个调闭包，链式法则自动展开
"""


class Value:
    """ 存储一个标量及其梯度 """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
