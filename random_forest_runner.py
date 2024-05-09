import multiprocessing
import sys

import matplotlib.pyplot as plt
import numpy as np

from dcs import RandomForestClassifier
from utils import eval, load_data, split_folds, split_train_test


def eval_one_fold(args):
    X, y, folds, k, classifier = args
    # Training and testing
    X_train, X_test, y_train, y_test = split_train_test(X, y, folds, k)
    print(f"Fitting k = {k}")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = eval(y_test, y_pred)
    print(f"Fit k = {k} done")
    return (metrics["accuracy"], metrics["f1"])


if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
        criterion = sys.argv[2]
        min_size_split = int(sys.argv[3])
        min_gain = float(sys.argv[4])
        max_depth = int(sys.argv[5])
    except Exception as err:
        print("Wrong arguments provided")
        print(f"{err}")
        exit()

    # Attribute type, True is categorical, False is numerical
    attr_type = None
    if "titanic" in dataset:
        attr_type = [True, True, False, False, False, False]
    elif "loan" in dataset:
        attr_type = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
        ]
    elif "parkinsons" in dataset:
        attr_type = [True for _ in range(22)]
    elif "digit" in dataset:
        attr_type = [True for _ in range(64)]

    if attr_type is None:
        print("Invalid dataset")
        exit()

    # Evaluate Random Forest on the datasets with these parameters
    X, y = load_data(dataset)
    folds = split_folds(y, 10)
    ntrees = [1, 5, 10, 20, 30, 40, 50]
    accuracy, f1 = [], []
    for ntree in ntrees:
        print(f"Fitting ntree = {ntree}")
        classifier = RandomForestClassifier(
            ntree=ntree,
            min_size_split=min_size_split,
            min_gain=min_gain,
            max_depth=max_depth,
            attr_type=attr_type,
            criterion=criterion,
        )
        accuracy_t, f1_t = [], []
        with multiprocessing.Pool(processes=5) as pool:
            results = pool.map(
                eval_one_fold,
                [(X, y, folds, k, classifier) for k in range(10)],
            )
            for i in range(10):
                accuracy_t.append(results[0])
                f1_t.append(results[1])

        accuracy.append(np.mean(accuracy_t))
        f1.append(np.mean(f1_t))

    f, axes = plt.subplots(1, 2, layout="constrained")
    axes[0].set_title("Accuracy vs ntree")
    axes[0].set_xlabel("ntree")
    axes[0].set_ylabel("Accuracy")
    axes[0].plot(ntrees, accuracy, marker="o")

    axes[1].set_title("F1 vs ntree")
    axes[1].set_xlabel("ntree")
    axes[1].set_ylabel("F1")
    axes[1].plot(ntrees, f1, marker="o")

    plt.savefig(f"figures/forest_{dataset}_{criterion}.jpg")
    print(f"Random forest {dataset} {criterion} graph created")
