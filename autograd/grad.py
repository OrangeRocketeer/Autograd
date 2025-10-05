import math
from graphviz import Digraph

class ValueNode():
    def __init__(self, value, prev=(), op='', label=''):
        self.value = value
        self.prev = set(prev)
        self.op = op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f'Value: {self.value}'

    def __add__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        out = ValueNode(self.value + other.value, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        out = ValueNode(self.value - other.value, (self, other), '-')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += out.grad * (-1.0)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        out = ValueNode(self.value * other.value, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.value
            other.grad += self.value * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        out = ValueNode(self.value / other.value, (self, other), '/')
        def _backward():
            self.grad += out.grad * (1 / other.value)
            other.grad += out.grad * (-self.value / other.value**2)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        return other / self

    def tanh(self):
        out = ValueNode(math.tanh(self.value), (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - (math.tanh(self.value)**2))
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = ValueNode(self.value**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.value ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def relu(self):
        out = ValueNode(0 if self.value < 0 else self.value, (self,), 'ReLU')
        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | value %.4f | grad %.4f }" % (n.label, n.value, n.grad), shape='record')
        if n.op:
            dot.node(name=uid + n.op, label=n.op)
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
