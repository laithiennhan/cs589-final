import sys

import matplotlib.pyplot as plt
import numpy as np

from dcs import RandomForestClassifier
from utils import eval, load_data, split_folds, split_train_test


def eval_dataset(
    dataset,
    attr_type,
    criterion="ig",
    min_size_split=2,
    min_gain=0.0,
    max_depth=None,
):
    """Evaluate Random Forest on the datasets with these parameters"""
    X, y = load_data(dataset)
    folds = split_folds(y, 10)
    ntrees = [1, 5, 10, 20, 30, 40, 50]
    accuracy, f1 = [], []
    for ntree in ntrees:
        classifier = RandomForestClassifier(
            ntree=ntree,
            min_size_split=min_size_split,
            min_gain=min_gain,
            max_depth=max_depth,
        )
        accuracy_t, f1_t = [], []
        for k in range(10):
            X_train, X_test, y_train, y_test = split_train_test(X, y, folds, k)
            classifier.fit(X_train, y_train, attr_type, criterion)
            y_pred = classifier.predict(X_test)
            metrics = eval(y_test, y_pred)
            accuracy_t.append(metrics["accuracy"])
            f1_t.append(metrics["f1"])

        accuracy.append(np.mean(accuracy_t))
        f1.append(np.mean(f1_t))

    f, axes = plt.subplots(1, 2, layout="constrained")
    axes[0].set_title("Accuracy vs ntree")
    axes[0].set_xlabel("ntree")
    axes[0].set_ylabel("Accuracy")
    axes[0].plot(ntrees, accuracy)

    axes[1].set_title("F1 vs ntree")
    axes[1].set_xlabel("ntree")
    axes[1].set_ylabel("F1")
    axes[1].plot(ntrees, f1)

    plt.savefig(f"figures/forset_{dataset}_{criterion}.jpg")
    print(f"Random forest {dataset} {criterion} graph created")


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        criterion = sys.argv[2]
        min_size_split = int(sys.argv[3])
        min_gain = float(sys.argv[4])
        max_depth = int(sys.argv[5])
    except Exception as err:
        print("Wrong arguments provided")
        print(f"{err}")
        exit()

    attr_type = None
    # Attribute type, True is categorical, False is numerical
    if "titanic" in dataset_name:
        attr_type = [True, True, False, False, False, False]
    elif "loan" in dataset_name:
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
    elif "parkinsons" in dataset_name:
        attr_type = [True for _ in range(22)]
    elif "digit" in dataset_name:
        attr_type = [True for _ in range(64)]

    if attr_type is None:
        print("Invalid dataset")
        exit()

    eval_dataset(
        dataset_name, attr_type, criterion, min_size_split, min_gain, max_depth
    )
