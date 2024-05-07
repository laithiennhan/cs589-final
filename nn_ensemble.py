import multiprocessing
import sys

import numpy as np

from nn import NeuralNetwork
from utils import (
    bootstrap,
    eval,
    get_majority,
    load_data,
    split_folds,
    split_train_test,
)


class NeuralNetworkEnsemble:
    def __init__(self, n_estimator, input_size, output_size, class_names=None):
        """Set attributes for classifier"""
        self.n_estimator = n_estimator
        self.models = []
        self.input_size = input_size
        self.output_size = output_size
        self.class_names = (
            class_names if class_names is not None else np.arange(output_size)
        )

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_estimator):
            model = NeuralNetwork(
                self.input_size, [10], self.output_size, self.class_names
            )
            X_bstrap, y_bstrap = bootstrap(X, y)
            model.fit(X_bstrap, y_bstrap, alpha=6, ld=0.01, num_epoch=300)
            self.models.append(model)

    def predict_one(self, X):
        result = np.empty(self.n_estimator)
        # Calculate the majority
        for i in range(self.n_estimator):
            result[i] = self.models[i].predict_one(X)

        return get_majority(result)

    def predict(self, X):
        return [self.predict_one(e) for e in X]


def eval_one_fold(args):
    X, y, folds, k, classifier = args
    X_train, X_test, y_train, y_test = split_train_test(X, y, folds, k)
    print(f"Fitting k = {k}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = eval(y_test, y_pred)
    print(f"Fit k = {k} done")
    return (metrics["accuracy"], metrics["f1"])


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
    except Exception as err:
        print("Wrong arguments provided")
        print(f"{err}")
        exit()

    # Load data
    output_size = 2
    if "digit" in dataset_name:
        output_size = 10
    elif "cleveland" in dataset_name:
        output_size = 2
    else:
        output_size = 2

    X, y = load_data(dataset_name, encode=True)
    folds = split_folds(y, 10)

    # Number of estimator in one ensemble
    n_estimator = 3

    # Training and testing
    classifier = NeuralNetworkEnsemble(3, X.shape[1], output_size, np.unique(y))
    accuracy_t, f1_t = [], []
    with multiprocessing.Pool() as pool:
        results = pool.map(
            eval_one_fold, [(X, y, folds, k, classifier) for k in range(10)]
        )
        for i in range(10):
            accuracy_t.append(results[0])
            f1_t.append(results[1])

    print(f"Num estimators: {n_estimator}")
    print(f"Accuracy: {np.mean(accuracy_t):.6f}")
    print(f"F1 Score: {np.mean(f1_t):.6f}")
