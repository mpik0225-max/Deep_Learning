import numpy as np

# Each Node object is one node of the computational graph.
class Node:

    ### Initialization of the class ###
    '''
    scalar: value of the node -- only scalars are used in this exercise
    _children: input nodes of the operation that created this node (a leading _ marks internal variables)
    _op: operation that created this node
    name: label used when printing/drawing the node
    '''
    def __init__(self, scalar, _children=(), _op='', name=''):
        self.scalar = scalar
        self.grad = 0.0 # By default we assume the variable does not affect the function
        self.name = name
        self._backward = lambda: None # Meaning the function doesn't do anything by default
        self._prev = set(_children) # Save the input nodes in a set.
        self._op = _op

    # This function just makes the printing of the Node human-readable
    def __repr__(self):
        return f"Node(name={self.name}, scalar={self.scalar}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.scalar + other.scalar, (self, other), '+') # forward pass

        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.scalar * other.scalar, (self, other), '*')

        def _backward():
            self.grad  += other.scalar * out.grad
            other.grad += self.scalar  * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # Only constant exponents (int/float) are supported, e.g. x**2 or x**-1
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Node(self.scalar**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.scalar**(other - 1) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.scalar
        out = Node(np.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.scalar * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.scalar
        t = 1 / (1 + np.exp(-x))
        out = Node(t, (self, ), 'sigmoid')

        def _backward():
            self.grad += t * (1 - t) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.scalar
        t = 2 / (1 + np.exp(-2*x)) - 1
        out = Node(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Node(max(0.0, self.scalar), (self, ), 'ReLU')

        def _backward():
            self.grad += (out.scalar > 0) * out.grad
        out._backward = _backward

        return out

    def log(self):
        assert self.scalar > 0, "log is only defined for positive numbers"
        x = self.scalar
        out = Node(np.log(x), (self, ), 'log')

        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward

        return out

    def sqrt(self):
        assert self.scalar > 0, "No complex numbers here"
        x = self.scalar
        out = Node(np.sqrt(x), (self, ), 'sqrt')

        def _backward():
            self.grad += 0.5 * x**-0.5 * out.grad
        out._backward = _backward

        return out

    # This function performs backpropagation for each node
    # Nothing to implement here but you can spend some time to understand how
    # the algorithm sorts the nodes.
    def backward(self):
        # topological order of all nodes in the graph (inputs before outputs)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # The gradient of the last operation wrt. itself is always 1.
        # We set this first gradient here.
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # Below are some necessary trivial definitions.
    # Feel free to use these functions. No gradient computation
    # is needed as they are implicitly implemented in prev. functions.
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------

def trace_nodes(node):
    '''
    Builds a list of all nodes of a graph. The method is
    only for inspection/debugging purposes. Feel free to use it if you need it.
    '''
    nodes = list()
    def build(v):
        if v not in nodes:
            nodes.insert(0, v)
            for child in (v._prev):
                build(child)
    build(node)
    return nodes

def print_graph(root):
    '''
    Produces a print out of the computational graph.
    '''
    nodes = trace_nodes(root)
    for n in nodes:
        print(n)
