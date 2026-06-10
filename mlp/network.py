import numpy as np
from mlp.activations import relu, relu_derivative, softmax


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

    def backward(self, y_true):
        N = y_true.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # Gradiente combinado softmax + cross-entropy: ŷ - y
        delta = self.a_values[-1] - y_true

        for i in reversed(range(self.num_layers)):
            grads_W[i] = self.a_values[i].T @ delta / N
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                # Regra da cadeia: propaga delta para a camada anterior
                # @ propaga pelo peso, * aplica derivada da ReLU element-wise
                delta = delta @ self.weights[i].T * relu_derivative(self.z_values[i - 1])

        return grads_W, grads_b

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
