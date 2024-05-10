import multiprocessing
import sys

import numpy as np

from nn import NeuralNetwork
from utils import eval, load_data, normalize, split_folds, split_train_test

num_epoch = 1000
batch_size = 32


def eval_one_fold(args):
    X, y, folds, k, classifier, ld, alpha = args
    # Training and testing
    X_train, X_test, y_train, y_test = split_train_test(X, y, folds, k)
    print(f"Fitting k = {k}")
    classifier.fit(
        X_train,
        y_train,
        alpha=alpha,
        ld=ld,
        batch_size=batch_size,
        num_epoch=num_epoch,
    )
    y_pred = classifier.predict(X_test)
    metrics = eval(y_test, y_pred)
    print(f"Fit k = {k} done")
    return (metrics["accuracy"], metrics["f1"])


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
    # Stratification
    folds = split_folds(y, 10)

    accuracy_t, f1_t = [], []
    print(f"Lambda = {ld}")
    print(f"Alpha = {alpha}")
    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(
            eval_one_fold, [(X, y, folds, k, classifier, ld, alpha) for k in range(10)]
        )
        for i in range(10):
            accuracy_t.append(results[0])
            f1_t.append(results[1])

    print(f"Neural Network architecture: {classifier.num_neurons}")
    print(f"Accuracy: {np.mean(accuracy_t):.6f}")
    print(f"F1 Score: {np.mean(f1_t):.6f}")
