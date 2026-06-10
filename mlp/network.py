import numpy as np
from mlp.activations import relu, softmax


class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]

        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)

            if i == self.num_layers - 1:
                A = softmax(Z)
            else:
                A = relu(Z)

            self.a_values.append(A)

        return A

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
