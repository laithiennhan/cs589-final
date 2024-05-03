import math

import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        num_neurons,
        output_size,
        class_names=None,
        init_value="uniform",
    ):
        self.weights = []
        self.gradients = []
        self.cost = 0
        self.prev_cost = 0
        self.pen = 0
        self.class_names = class_names if class_names is not None else [1]
        self.input_size = input_size
        self.num_neurons = np.append(num_neurons, output_size)
        self.num_neurons = np.insert(self.num_neurons, 0, input_size)
        self.output_size = output_size
        self.init_value = init_value
        self.cost_arr = []

    def init_weights(self):
        # Init weights of NN
        self.weights = []
        self.gradients = []
        self.cost = 0
        self.prev_cost = 0
        self.pen = 0
        num_neurons = self.num_neurons
        for i in range(1, len(num_neurons)):
            self.weights.append(
                np.array(
                    [
                        [
                            (
                                np.random.uniform(-1, 1)
                                if self.init_value == "uniform"
                                else np.random.normal(0, 1)
                            )
                            for _ in range(num_neurons[i - 1] + 1)
                        ]
                        for _ in range(num_neurons[i])
                    ]
                )
            )
            self.gradients.append(
                np.array(
                    [
                        [0 for _ in range(num_neurons[i - 1] + 1)]
                        for _ in range(num_neurons[i])
                    ],
                    dtype=float,
                )
            )

    def fit(
        self,
        X,
        y,
        alpha=1.0,
        x_test=[],
        y_test=[],
        ld=0.1,
        num_epoch=1000,
        batch_size=32,
        theta=None,
    ):
        n_row = X.shape[0]
        self.init_weights()

        if theta is not None:
            self.weights = theta

        y_vectors = []
        for i in range(len(y)):
            y_vector = (self.class_names == y[i]).astype(int)
            y_vectors.append(y_vector)

        for epoch in range(num_epoch):
            if epoch > 10 and epoch % 20 == 0:
                if self.prev_cost - self.cost < 1e-5:
                    if alpha > 1e-4:
                        alpha *= 0.5
                    else:
                        print(f"Traning stopped at epoch {epoch}")
                        break

            self.prev_cost = self.cost
            self.cost = 0
            self.pen = 0

            for i in range(math.ceil(n_row / batch_size)):
                batch_X, batch_y = [], []

                batch_X = X[i * batch_size : min((i + 1) * batch_size, n_row)]
                batch_y = y_vectors[i * batch_size : min((i + 1) * batch_size, n_row)]

                # Forward propagation
                activations, _ = self.forward_propagate(batch_X, batch_y)

                # Backward propagation
                self.backward_propagate(activations, batch_y)

                # Apply gradient
                self.apply_gradient(
                    alpha,
                    ld,
                    min((i + 1) * batch_size, n_row) - batch_size * i,
                    x_test,
                    y_test,
                )

            self.cost /= n_row
            self.pen *= ld / (2 * n_row)
            self.cost += self.pen

    def forward_propagate(self, X, y_vectors):
        res = []
        cost = 0
        for i in range(len(X)):
            activation = self.forward_propagate_one(X[i])
            y = y_vectors[i]
            # Caculate cost J
            j = sum(
                [
                    (
                        -y[i] * np.log(activation[-1][i])
                        - (1 - y[i]) * (np.log(1 - activation[-1][i]))
                    )
                    for i in range(len(y))
                ]
            )
            cost += j
            res.append(activation)

        self.cost += cost
        return res, cost / len(X)

    def backward_propagate(self, activations, y_vectors):
        for i in range(len(activations)):
            activation = activations[i]
            y = y_vectors[i]
            cur_delta = activation[-1] - y

            for j in range(len(activation) - 1, 0, -1):
                # Calculate this layer gradients
                gradients = np.outer(cur_delta, activation[j - 1])

                self.gradients[j - 1] += gradients

                # Calculate layer delta
                temp = activation[j - 1]
                cur_delta = (
                    temp
                    * (1 - temp)
                    * np.dot(np.transpose(self.weights[j - 1]), cur_delta)
                )
                cur_delta = np.delete(
                    cur_delta, 0
                )  # Remove bias because delta values do not need

    def apply_gradient(self, alpha, ld, batch_size, x_test=[], y_test=[]):
        for i in range(len(self.weights) - 1, -1, -1):
            # Add penalty
            w = self.weights[i].copy()
            w[:, 0] = 0
            self.pen += np.sum(np.square(w))
            self.gradients[i] += ld * w
            self.gradients[i] /= batch_size
            # Update weights
            self.weights[i] -= alpha * self.gradients[i]
            self.gradients[i].fill(0)

        if len(x_test) != 0:
            y_vectors = []
            for i in range(len(y_test)):
                y_vector = (self.class_names == y_test[i]).astype(int)
                y_vectors.append(y_vector)

            _, cost = self.forward_propagate(x_test, y_vectors)
            self.cost_arr.append(cost)

    def forward_propagate_one(self, X):
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        activation = [X.copy()]
        for i, matrix in enumerate(self.weights):
            # Add bias for gradient calculation
            activation[-1] = np.insert(activation[-1], 0, 1)
            z = np.dot(matrix, activation[-1])
            next_layer = sigmoid(z)
            activation.append(next_layer)

        return activation

    def predict_one(self, X):
        output = self.forward_propagate_one(X)[-1]
        return self.class_names[np.argmax(output)]

    def predict(self, X):
        return [self.predict_one(e) for e in X]

    def plot(self, fig_name, step_size):
        plt.clf()
        plt.plot(np.arange(len(self.cost_arr)) * step_size, self.cost_arr)
        plt.title("Performance J on test set")
        plt.xlabel("Number of training samples given to the network")
        plt.ylabel("Performance (J)")
        plt.savefig(fig_name)
