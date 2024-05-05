import csv
import random

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def entropy(data):
    """Calculate entropy"""
    _, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities))


def gini(data):
    """Calculate gini"""
    _, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / np.sum(counts)
    return 1 - np.sum(probabilities**2)


def gini_gain(mp, N):
    """Calculate information gain"""
    avg_gini = 0
    for v in mp.keys():
        avg_gini += len(mp[v]) / N * gini(np.array(mp[v]))
    return avg_gini


def information_gain(mp, N, initial_entropy):
    """Calculate information gain"""
    avg_entropy = 0
    for v in mp.keys():
        avg_entropy += len(mp[v]) / N * entropy(np.array(mp[v]))
    return initial_entropy - avg_entropy


def probability(data, v):
    """Calculate probability of v in data"""
    label, counts = np.unique(data[:, -1], return_counts=True)
    return counts[list(label).index(v)] / np.sum(counts)


def bootstrap(X, y):
    """Return a bootstrap of X, y"""
    index = np.arange(len(X))
    index = random.choices(index, k=len(X))
    return np.array(X)[index], np.array(y)[index]


def split_folds(y, k):
    """Split data into k folds for stratified cross validation"""
    unique_class = np.unique(y)
    class_bins = {cls: [] for cls in unique_class}

    # Split indices into sets based on class
    for i, cls in enumerate(y):
        class_bins[cls].append(i)

    # Split into k folds
    for cls in class_bins:
        class_bins[cls] = np.array_split(class_bins[cls], k)

    # Generate folds
    folds = []
    for i in range(k):
        fold = []
        for cls in class_bins:
            fold.extend(class_bins[cls][i])
        folds.append(fold)

    return folds


def split_train_test(X, y, folds, k):
    """Split folds into train and test set"""
    # Combine the train folds indices
    train_indices = folds.copy()
    train_indices.pop(k)
    train_indices = sum(train_indices, [])

    # Shuffle train set
    np.random.shuffle(train_indices)

    return (
        np.array(X[train_indices]),
        np.array(X[folds[k]]),
        np.array(y[train_indices]),
        np.array(y[folds[k]]),
    )


def eval(y_true, y_pred):
    """Compute the evaluation metrics"""
    res = {}
    unique_class = np.unique(y_true)
    accuracy, precision, recall, f1 = (
        np.zeros(unique_class.shape[0]),
        np.zeros(unique_class.shape[0]),
        np.zeros(unique_class.shape[0]),
        np.zeros(unique_class.shape[0]),
    )
    matrix = np.zeros((unique_class.shape[0], unique_class.shape[0]), dtype=int)

    for i in range(len(y_pred)):
        matrix[
            list(unique_class).index(y_true[i]),
            list(unique_class).index(y_pred[i]),
        ] += 1

    for i in range(len(unique_class)):
        accuracy[i] = np.trace(matrix) / len(y_true)
        if np.sum(matrix[:, i]) != 0:
            # Cases when predict positive are 0
            precision[i] = matrix[i, i] / np.sum(matrix[:, i])
        if np.sum(matrix[i]) != 0:
            recall[i] = matrix[i, i] / np.sum(matrix[i])

    f1 = []
    for i in range(len(precision)):
        if precision[i] + recall[i] == 0:
            continue
        else:
            f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))

    f1 = np.mean(f1)
    accuracy = np.mean(accuracy)
    precision = np.mean(precision)
    recall = np.mean(recall)

    res["class"] = unique_class
    res["matrix"] = matrix
    res["accuracy"] = accuracy
    res["precision"] = precision
    res["recall"] = recall
    res["f1"] = f1
    return res


def get_majority(data):
    """Return the majority of a list"""
    unique_vals, counts = np.unique(data, return_counts=True)
    max_indices = np.argwhere(counts == np.max(counts))
    return unique_vals[random.choice(max_indices)]


def one_hot_encode(data, column):
    """One hot encoder, column is a list of index of column to encode"""
    transformer = make_column_transformer(
        (OneHotEncoder(), column), remainder="passthrough"
    )
    transformed = transformer.fit_transform(data)
    return np.array(transformed)


def normalize(data):
    """Normalize data by column"""
    scaler = MinMaxScaler()
    return np.array(scaler.fit_transform(data))


def convert_to_float(item):
    try:
        return float(item)
    except ValueError:
        return item


def ordinal_encoding(dataset, categorical_indices):
    enc = OrdinalEncoder()
    dataset[:, categorical_indices] = enc.fit_transform(dataset[:, categorical_indices])


def load_data(filename, encode=False):
    data = []
    label = []

    if filename == "loan.csv":
        with open("datasets/loan.csv", "r") as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                features = [convert_to_float(item) for item in line[:-1]]
                if line[-1] == "Y":
                    label.append(0)
                else:
                    label.append(1)
                data.append(features)
        # Remove loan id
        data = np.delete(data, 0, axis=1)
        ordinal_encoding(data, [2])
        if encode:
            data = one_hot_encode(data, [0, 1, 3, 4, 9, 10])
        else:
            ordinal_encoding(data, [0, 1, 3, 4, 9, 10])

    elif filename == "titanic.csv":
        with open("datasets/titanic.csv", "r") as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                features = [convert_to_float(item) for item in line[1:]]
                label.append(line[0])
                data.append(features)

        # Remove name attributes
        data = np.delete(data, 1, axis=1)
        if encode:
            data = one_hot_encode(data, [1])
        else:
            ordinal_encoding(data, [1])
    elif filename == "digits":
        data, label = load_digits(return_X_y=True)
    elif filename == "cleveland.csv":
        with open("datasets/cleveland.csv", "r") as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                if "?" in line:
                    continue
                features = [convert_to_float(item) for item in line[:-1]]
                label.append(line[-1])
                data.append(features)
    else:
        with open(f"datasets/{filename}", "r") as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                features = [convert_to_float(item) for item in line[:-1]]
                label.append(line[-1])
                data.append(features)
        data = np.array(data)

    return np.array(data, dtype=float), np.array(label, dtype=float)
