import sys

import numpy as np

from nn import NeuralNetwork
from utils import load_data, normalize


def split_train_test(X, y):
    k = int(0.8 * len(X))
    index = np.random.permutation(np.arange(len(X)))
    return X[index[:k]], X[index[k:]], y[index[:k]], y[index[k:]]


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        ld = float(sys.argv[2])
        alpha = float(sys.argv[3])
        num_hidden_layers = int(sys.argv[4])
        if len(sys.argv) < num_hidden_layers + 5:
            raise Exception()
        num_neurons = []
        for i in range(5, 5 + num_hidden_layers):
            num_neurons.append(int(sys.argv[i]))
    except Exception as err:
        print("Wrong arguments provided")
        print(f"{err}")
        exit()

    # Load data
    X, y = load_data(dataset_name, encode=True)
    output_size = 2
    if "digit" in dataset_name:
        output_size = 10
    elif "cleveland" in dataset_name:
        output_size = 5
    else:
        output_size = 2

    classifier = NeuralNetwork(
        X.shape[1], num_neurons, output_size, class_names=np.unique(y)
    )
    X = normalize(X)

    # Training and testing
    num_epoch = 1
    batch_size = 20

    accuracy_t, f1_t = [], []
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    classifier.fit(
        X_train,
        y_train,
        alpha=alpha,
        ld=ld,
        batch_size=batch_size,
        num_epoch=num_epoch,
        x_test=X_test,
        y_test=y_test,
    )
    y_pred = classifier.predict(X_test)

    classifier.plot(f"figures/nn_{dataset_name}_cost.png", batch_size)
    print("Graph J created")
