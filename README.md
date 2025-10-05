# Autograd Engine

A lightweight automatic differentiation engine built from scratch in Python. This implementation provides a computational graph framework with reverse-mode autodiff (backpropagation) and neural network building blocks.

## Overview

This project implements a minimal autograd engine capable of building and training neural networks. It demonstrates the fundamental concepts behind modern deep learning frameworks like PyTorch and TensorFlow, with a focus on clarity and educational value.

The engine supports automatic differentiation through computational graphs, allowing gradients to be computed efficiently via backpropagation. It includes implementations of basic neural network components including neurons, layers, and multi-layer perceptrons.

## Features

- **Automatic Differentiation**: Reverse-mode autodiff with dynamic computational graph construction
- **Scalar Operations**: Support for addition, subtraction, multiplication, division, and exponentiation
- **Activation Functions**: ReLU and hyperbolic tangent (tanh) implementations
- **Neural Network Primitives**: Modular neuron, layer, and MLP classes
- **Graph Visualization**: Built-in computational graph visualization using Graphviz
- **Training Utilities**: Gradient descent with learning rate scheduling and mini-batch support
- **Pure Python**: Minimal dependencies, easy to understand and modify

## Installation

### Prerequisites

```bash
pip install numpy graphviz matplotlib scikit-learn
```

### Setup

Clone the repository:
```bash
git clone https://github.com/yourusername/autograd-engine.git
cd autograd-engine
```

Install the package:
```bash
pip install -e .
```

## Quick Start

### Basic Operations

```python
from autograd.grad import ValueNode

# Create scalar values
a = ValueNode(2.0, label='a')
b = ValueNode(-3.0, label='b')
c = ValueNode(10.0, label='c')

# Build computational graph
d = a * b + c
e = d.relu()

# Compute gradients
e.backward()

print(f"e.value = {e.value}")  # Forward pass result
print(f"a.grad = {a.grad}")    # Gradient of e with respect to a
```

### Neural Network Example

```python
from autograd.nn import MLP
from autograd.grad import ValueNode
import numpy as np

# Create a multi-layer perceptron
model = MLP(n_inputs=3, n_outputs=[4, 4, 1])

# Forward pass
x = [ValueNode(1.0), ValueNode(-2.0), ValueNode(3.0)]
output = model(x)

# Backward pass
output.backward()

# Access parameters and gradients
params = model.parameters()
print(f"Total parameters: {len(params)}")
```

### Training on Make Moons Dataset

```python
from sklearn.datasets import make_moons
from autograd.nn import MLP
from autograd.grad import ValueNode
import numpy as np

# Generate dataset
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert to -1, 1

# Initialize model
model = MLP(2, [16, 16, 1])

# Define loss function
def loss(batch_size=None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    
    inputs = [list(map(ValueNode, xrow)) for xrow in Xb]
    scores = list(map(model, inputs))
    
    # SVM max-margin loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    
    # Accuracy
    accuracy = [(yi > 0) == (scorei.value > 0) for yi, scorei in zip(yb, scores)]
    return data_loss, sum(accuracy) / len(accuracy)

# Train the model
model.train(loss, epochs=50, schedule=True)
```

## Architecture

### Core Components

**ValueNode**: The fundamental building block representing a node in the computational graph. Stores a scalar value, gradient, operation type, and parent nodes.

**Module**: Abstract base class providing parameter management and gradient zeroing functionality.

**Neuron**: Implements a single neuron with configurable activation (ReLU or linear).

**Layer**: Collection of neurons with shared input dimensionality.

**MLP**: Multi-layer perceptron combining multiple layers into a complete neural network.

### Computational Graph

The engine builds a dynamic computational graph during the forward pass. Each operation creates a new ValueNode with references to its inputs, forming a directed acyclic graph (DAG). The backward pass traverses this graph in reverse topological order, applying the chain rule to compute gradients efficiently.

## Mathematical Foundation

The engine implements reverse-mode automatic differentiation based on the chain rule:

```
∂L/∂x = Σᵢ (∂L/∂uᵢ) · (∂uᵢ/∂x)
```

Where L is the loss, x is an input variable, and uᵢ are intermediate variables that depend on x.

Each operation stores a local derivative function (`_backward`) that computes:
- How the operation's output changes with respect to each input (local derivative)
- Multiplies by the upstream gradient (chain rule)
- Accumulates into each input's gradient (supports multiple paths)

## Visualization

Visualize computational graphs using the built-in Graphviz integration:

```python
from autograd.grad import ValueNode, draw_dot

a = ValueNode(2.0, label='a')
b = ValueNode(-3.0, label='b')
c = a * b
c.backward()

# Generate graph visualization
graph = draw_dot(c)
graph.render('computation_graph', format='png')
```

## Performance

The implementation prioritizes clarity over performance. Benchmarks against micrograd and PyTorch show comparable accuracy on small-scale problems, with expected performance differences due to Python overhead and lack of vectorization.

**Make Moons Classification (100 samples, 2 features)**
- Final Training Accuracy: 100%
- Convergence: ~35-40 epochs
- Architecture: [2, 16, 16, 1]

## Limitations

- **Scalar Operations**: All operations work on scalars, not tensors (no vectorization)
- **Performance**: Pure Python implementation without optimization
- **Limited Operations**: Basic set of operations compared to production frameworks
- **No GPU Support**: CPU-only computation

## Inspiration

This project was inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). The implementation follows similar architectural principles while adding additional documentation and examples focused on understanding the underlying mathematics.

## Contributing

Contributions are welcome. Please ensure code follows the existing style and includes appropriate tests and documentation.

## License

MIT License - see LICENSE file for details

## Author

**Ayush Singh**  
B.Tech Mathematics and Computing, IIT Mandi  
ayush.rocketeer@gmail.com

## Acknowledgments

- Andrej Karpathy for micrograd and educational materials
- The PyTorch team for API design inspiration
- IIT Mandi for academic support

## References

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) video series
