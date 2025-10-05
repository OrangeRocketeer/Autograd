import numpy as np
from autograd.grad import ValueNode

class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_inputs, activate=True):
        self.w = [ValueNode(np.random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = ValueNode(np.random.uniform(-1,1))
        self.activate = activate

    def __call__(self, x):
        z = sum((w*x for w,x in zip(self.w, x)), self.b)
        return z.relu() if self.activate else z

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.activate else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, n_inputs, n_outputs):
        neuron_list = [n_inputs] + n_outputs
        self.layers = [Layer(neuron_list[i], neuron_list[i+1], activate=i!=len(n_outputs)-1) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def train(self, loss, epochs, schedule=True, batch_size=None):
        for epoch in range(epochs):
            total_loss, accuracy = loss(batch_size)
            self.zero_grad()
            total_loss.backward()
            learning_rate = (1 - 0.9*epoch/100) if schedule else 0.03
            for parameter in self.parameters():
                parameter.value -= learning_rate * parameter.grad
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.value:.5f}, Accuracy: {accuracy*100:.2f}%')
