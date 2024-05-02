import csv
import math
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn import datasets


def euclidean_distance(r1, r2):
    s = 0
    for i in range(0, len(r1) - 1):
        s += (r1[i] - r2[i]) ** 2
    e_distance = math.sqrt(s)
    return e_distance


def find_k_nearest_neighbors(r, k, training_set):
    distances = [[row, euclidean_distance(r, row)] for row in training_set]
    distances.sort(key=lambda x: x[1])
    return [distances[i][0] for i in range(k)]

# Ordinal encoding for categorical features


def ordinal_encoding(categorical_indices, dataset):
    dataset_copy = dataset.copy()
    for i in categorical_indices:
        curr_val = 0
        value_map = dict()
        for row in dataset_copy:
            if row[i] not in value_map.keys():
                row[i] = curr_val
                value_map[row[i]] = curr_val
                curr_val += 1
            else:
                row[i] = value_map[row[i]]
    print(dataset_copy)
    return dataset_copy


def convert_to_float(item):
    try:
        return float(item)
    except ValueError:
        return item


def load_data(filename):
    data = []

    if (filename == 'loan.csv'):
        with open('datasets/loan.csv', 'r') as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                features = [convert_to_float(item) for item in line[:-1]]
                label = line[-1]
                data.append(features + [label])
        data = [row[1:] for row in data]
        return ordinal_encoding([0, 1, 2, 3, 4, 9, 10], data)
    elif (filename == 'titanic.csv'):
        with open('datasets/titanic.csv', 'r') as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                features = [convert_to_float(item) for item in line[1:]]
                label = line[0]
                data.append(features + [label])
        data = [row[:1] + row[2:] for row in data]
        return ordinal_encoding([1], data)
    else:
        with open(f'datasets/{filename}', 'r') as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            for line in csvFile:
                features = [convert_to_float(item) for item in line[:-1]]
                label = line[-1]
                data.append(features + [label])
        return ordinal_encoding([], data)


def create_stratified_folds(data, k):
    class_dict = defaultdict(list)
    for item in data:
        class_dict[item[-1]].append(item)

    folds = [[] for _ in range(k)]
    for _, items in class_dict.items():
        np.random.shuffle(items)
        for idx, item in enumerate(items):
            folds[idx % k].append(item)

    return folds


def train_knn(k, training_set, test_set):
    correct_predictions = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for test_instance in test_set:
        neighbors = find_k_nearest_neighbors(test_instance, k, training_set)
        predicted_class = max(set([neighbor[-1] for neighbor in neighbors]),
                              key=[neighbor[-1] for neighbor in neighbors].count)
        if predicted_class == test_instance[-1]:
            correct_predictions += 1
    return correct_predictions / len(test_set)


def normalize_datasets(training_set, testing_set):
    new_training = training_set.copy()
    new_testing = testing_set.copy()
    min_col = [0] * (len(training_set[0]) - 1)
    max_col = [0] * (len(training_set[0]) - 1)
    for i in range(len(min_col)):
        min_col[i] = min([row[i] for row in training_set])
        max_col[i] = max([row[i] for row in training_set])
    for i in range(len(training_set)):
        for j in range(len(training_set[0]) - 1):
            new_training[i][j] = (new_training[i][j] - min_col[j]
                                  ) / (max_col[j] - min_col[j])
    for i in range(len(testing_set)):
        for j in range(len(testing_set[0]) - 1):
            new_testing[i][j] = (new_testing[i][j] - min_col[j]
                                 ) / (max_col[j] - min_col[j])
    return new_training, new_testing


filename = 'digits'

if (filename == "digits"):
    digits = datasets.load_digits()
    data = digits.data
else:
    data = load_data(f"datasets/{filename}")

k = 10
folds = create_stratified_folds(data, k)

k_values = range(1, 52, 2)
accuracies = {k_val: [] for k_val in k_values}

for k_val in k_values:
    fold_accuracies = []
    for i in range(k):
        test_set = folds[i]
        training_set = [item for j in range(k) if j != i for item in folds[j]]
        if (filename != 'digits'):
            training_set, test_set = normalize_datasets(training_set, test_set)
        accuracy = train_knn(k_val, training_set, test_set)
        fold_accuracies.append(accuracy)
    accuracies[k_val].append(np.mean(fold_accuracies))

ks = list(k_values)
mean_accuracies = [np.mean(accuracies[k]) for k in ks]

plt.figure(figsize=(10, 5))
plt.plot(ks, mean_accuracies, marker='o')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Cross-validated Accuracy')
plt.title(
    f'KNN Stratified Cross-Validation Performance with feature normalization for {filename} dataset')
plt.grid(True)
plt.show()
