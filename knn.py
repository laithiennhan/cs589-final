from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from utils import convert_to_float, load_data, normalize, split_folds, split_train_test, eval, get_majority
import sys


def euclidean_distance(r1, r2):
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    return np.sqrt(np.sum((r1 - r2) ** 2))


def find_k_nearest_neighbors(r, k, training_X, training_Y):
    distances = np.array([])
    for i in range(len(training_X)):
        distances = np.append(distances, euclidean_distance(r, training_X[i]))
    nearest_indices = np.argsort(distances)[:k]

    return [training_Y[i] for i in nearest_indices]


def train_knn(k_val, data, is_normalize):
    # correct_predictions = 0
    # tp = defaultdict(int)
    # fp = defaultdict(int)
    # fn = defaultdict(int)
    (training_X, testing_X, training_y, testing_y) = data
    y_pred = []
    y_true = []
    training_X = np.array(training_X, dtype=float)
    testing_X = np.array(testing_X, dtype=float)
    if is_normalize:
        normalize(training_X)
        normalize(testing_X)
    for i in range(len(testing_X)):
        neighbors = find_k_nearest_neighbors(
            testing_X[i], k_val, training_X, training_y)
        predicted_class = get_majority(neighbors)
        y_pred.append(predicted_class)
        y_true.append(testing_y[i])
    return eval(y_true, y_pred)


def parallel_knn(k_values, X, y, k, is_normalize=False):
    results = {k_val: {'accuracies': [], 'f1_scores': []}
               for k_val in k_values}

    for k_val in k_values:
        folds = split_folds(y, k)

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_fold = {executor.submit(
                train_knn, k_val, split_train_test(X, y, folds, i), is_normalize): i for i in range(k)}

            for future in as_completed(future_to_fold):
                fold_index = future_to_fold[future]

                train_results = future.result()
                results[k_val]['accuracies'].append(
                    train_results["accuracy"])
                results[k_val]['f1_scores'].append(train_results["f1"])

        # Calculate mean accuracy for this k value
        results[k_val]['mean_accuracy'] = np.mean(results[k_val]['accuracies'])
        results[k_val]['mean_f1_score'] = np.mean(results[k_val]['f1_scores'])

    return results


def main(filename, is_normalize=False):

    X, y = load_data(filename, encode=True)
    k = 10
    k_values = range(1, 42, 2)
    results = parallel_knn(k_values, X, y, k, is_normalize)

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


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print("Usage: python main.py <filename> (optional: use flag --normalize to enable data normalization)")
        sys.exit(1)
    if len(sys.argv) == 3:
        if sys.argv[2] == '--normalize':
            main(sys.argv[1], is_normalize=True)
        else:
            print(
                "Usage: python main.py <filename> (optional: use flag --normalize to enable data normalization)")
        sys.exit(1)
    else:
        main(sys.argv[1])
