from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn import datasets


def euclidean_distance(r1, r2):
    return np.sqrt(np.sum((np.array(r1[:-1]) - np.array(r2[:-1])) ** 2))


def find_k_nearest_neighbors(r, k, training_set):
    distances = np.array([euclidean_distance(r, row) for row in training_set])
    nearest_indices = np.argsort(distances)[:k]
    return [training_set[i] for i in nearest_indices]


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
    labels = []
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
                label = float(line[-1])
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
                                  ) / (max_col[j] - min_col[j]) if max_col[j] - min_col[j] != 0 else 0
    for i in range(len(testing_set)):
        for j in range(len(testing_set[0]) - 1):
            new_testing[i][j] = (new_testing[i][j] - min_col[j]
                                 ) / (max_col[j] - min_col[j]) if max_col[j] - min_col[j] != 0 else 0
    return new_training, new_testing


def train_knn(k, training_set, test_set, normalize):
    correct_predictions = 0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # for label in labels:
    #     tp[label] = 0
    #     fp[label] = 0
    #     tn[label] = 0
    #     fn[label] = 0

    if normalize:
        training_set, test_set = normalize_datasets(training_set, test_set)
    for test_instance in test_set:
        neighbors = find_k_nearest_neighbors(test_instance, k, training_set)
        predicted_class = max(set([neighbor[-1] for neighbor in neighbors]),
                              key=[neighbor[-1] for neighbor in neighbors].count)
        if predicted_class == test_instance[-1]:
            correct_predictions += 1
            tp[predicted_class] += 1
        else:
            fp[predicted_class] += 1
            fn[test_instance[-1]] += 1

    accuracy = correct_predictions / len(test_set)
    f1_scores = []
    # Confusion matrix
    labels = set(tp.keys()).union(
        fp.keys()).union(fp.keys())
    for label in labels:
        f1_scores.append(tp[label] / (tp[label] + 1 /
                         2 * (fp[label] + fn[label])))

    return accuracy, np.mean(f1_scores)


def parallel_knn(k_values, folds, k, normalize=False):
    results = {k_val: {'accuracies': [], 'f1_scores': []}
               for k_val in k_values}

    for k_val in k_values:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_fold = {executor.submit(train_knn, k_val, [item for j in range(
                k) if j != i for item in folds[j]], folds[i], normalize): i for i in range(k)}

            for future in as_completed(future_to_fold):
                fold_index = future_to_fold[future]
                try:
                    accuracy, f1_score = future.result()
                    results[k_val]['accuracies'].append(accuracy)
                    results[k_val]['f1_scores'].append(f1_score)
                except Exception as e:
                    print(f'Fold {fold_index} generated an exception: {e}')

        # Calculate mean accuracy for this k value
        results[k_val]['mean_accuracy'] = np.mean(results[k_val]['accuracies'])
        results[k_val]['mean_f1_score'] = np.mean(results[k_val]['f1_scores'])

    return results


filename = 'loan.csv'

data = []
labels = []

if (filename == "titanic.csv"):
    digits = datasets.load_digits()
    data = digits.data

else:
    data = load_data(filename)


k = 10
folds = create_stratified_folds(data, k)

k_values = range(1, 40, 2)
results = parallel_knn(k_values, folds, k, normalize=True)

ks = list(k_values)


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(ks, [results[k]['mean_accuracy'] for k in ks], marker='o')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Cross-validated Accuracy')
plt.title(f'KNN Accuracy {filename}')

plt.subplot(1, 2, 2)
plt.plot(ks, [results[k]['mean_f1_score'] for k in ks], marker='o')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Cross-validated F1 Score')
plt.title(f'KNN F1 Score {filename}')

plt.tight_layout()
plt.savefig(f'./figures/knn_{filename}.png')
