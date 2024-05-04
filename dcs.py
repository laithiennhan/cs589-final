import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils import (
    bootstrap,
    entropy,
    get_majority,
    gini,
    gini_gain,
    information_gain,
    probability,
)


# Node Interface
class INode:
    def predict(self, entry):
        raise NotImplementedError


# Decision Node
class DNode(INode):
    def __init__(self, attr_name, child, original, is_categorical, thres):
        self.attr_name = attr_name
        self.child = child
        self.original = original
        self.is_categorical = is_categorical
        self.thres = thres

    def predict(self, entry):
        if self.is_categorical:
            if entry[self.attr_name] not in self.child:
                unique_class, counts = np.unique(
                    self.original[:, -1], return_counts=True
                )
                return unique_class[counts.argmax()]
            else:
                return self.child[entry[self.attr_name]].predict(entry)
        else:
            if entry[self.attr_name] > self.thres:
                return self.child["greater"].predict(entry)
            else:
                return self.child["fewer"].predict(entry)


# Leaf Node
class LNode(INode):
    def __init__(self, label):
        self.label = label

    def predict(self, entry):
        return self.label


def split_attr_cat(data, attr, criterion="ig", information=None):
    """Split categorical attributes"""
    if criterion == "ig" and not information:
        raise Exception("information is None")
    mp = {}
    for e in data:
        if e[attr] not in mp:
            mp[e[attr]] = [e]
        else:
            mp[e[attr]].append(e)

    if criterion == "gini":
        gain = gini_gain(mp, len(data))
    else:
        gain = information_gain(mp, len(data), information)
    return (mp, gain)


def split_attr_num(data, attr, criterion="ig", information=None):
    """Split numerical attributes"""
    if criterion == "ig" and not information:
        raise Exception("information is None")
    temp = np.array(sorted(data, key=lambda x: x[attr]))
    best_thres = np.mean([temp[0, attr], temp[1, attr]])
    max_gain_ig = 0
    max_gain_gini = gini(data)
    best_split = {}
    for i in range(temp.shape[0] - 1):
        mp = {}
        mp["fewer"] = temp[: i + 1]
        mp["greater"] = temp[i:]
        if criterion == "gini":
            gain = gini_gain(mp, len(data))
            if gain < max_gain_gini:
                max_gain_gini = gain
                best_thres = np.mean([temp[i, attr], temp[i + 1, attr]])
                best_split = mp
        else:
            gain = information_gain(mp, data.shape[0], information)
            if gain > max_gain_ig:
                max_gain_ig = gain
                best_thres = np.mean([temp[i, attr], temp[i + 1, attr]])
                best_split = mp

    return best_split, max_gain_gini if criterion == "gini" else max_gain_ig, best_thres


class DecisionTreeClassifier:
    def __init__(self, min_size_split=2, min_gain=0.0, max_depth=None):
        """Set attributes for classifier"""
        self.min_size_split = min_size_split
        self.min_gain = min_gain
        self.max_depth = max_depth

    def fit(self, X, y, attr_type, criterion="ig"):
        self.num_attr = X.shape[1]
        # Establish the attributes type for every feature
        self.attr_type = attr_type

        if criterion == "gini":
            self.root = self.build_tree_gini(
                np.append(X, y[:, np.newaxis], axis=1), list(range(X.shape[1])), 0
            )
        else:
            self.root = self.build_tree(
                np.append(X, y[:, np.newaxis], axis=1), list(range(X.shape[1])), 0
            )

    def build_tree(self, data, attributes, depth):
        """Information gain criterion"""
        information = entropy(data)
        if information == 0 or data.shape[0] < self.min_size_split:
            return LNode(get_majority(data[:, -1]))
        if self.max_depth and depth > self.max_depth:
            return LNode(get_majority(data[:, -1]))

        for v in np.unique(data[:, -1]):
            if probability(data, v) > 0.85:
                return LNode(get_majority(data[:, -1]))

        max_gain = 0
        best_attr = -1
        best_split = {}
        is_categorical = True
        best_thres = None
        # Getting a subset of attributes to test split on
        test_attr = random.sample(attributes, int(np.sqrt(self.num_attr)))
        for attr in test_attr:
            if self.attr_type[attr]:
                thres = None
                mp, gain = split_attr_cat(data, attr, "ig", information)
            else:
                mp, gain, thres = split_attr_num(data, attr, "ig", information)

            if gain > max_gain:
                max_gain = gain
                best_attr = attr
                best_split = mp
                is_categorical = self.attr_type[attr]
                best_thres = thres

        if max_gain <= self.min_gain:
            return LNode(get_majority(data[:, -1]))
        child = {}
        for v, s in best_split.items():
            child[v] = self.build_tree(np.array(s), attributes, depth + 1)
        return DNode(best_attr, child, data, is_categorical, best_thres)

    def build_tree_gini(self, data, attributes, depth):
        """Gini criterion"""
        if data.shape[0] < self.min_size_split:
            return LNode(get_majority(data[:, -1]))
        if self.max_depth and depth > self.max_depth:
            return LNode(get_majority(data[:, -1]))

        for v in np.unique(data[:, -1]):
            if probability(data, v) > 0.85:
                return LNode(get_majority(data[:, -1]))

        original_gini = gini(data)
        max_gain = original_gini
        best_attr = -1
        best_split = {}
        is_categorical = True
        best_thres = None
        # Getting a subset of attributes to test split on
        test_attr = random.sample(attributes, int(np.sqrt(self.num_attr)))
        for attr in test_attr:
            if self.attr_type[attr]:
                thres = None
                mp, gain = split_attr_cat(data, attr, "gini")
            else:
                mp, gain, thres = split_attr_num(data, attr, "gini")

            if gain < max_gain:
                max_gain = gain
                best_attr = attr
                best_split = mp
                is_categorical = self.attr_type[attr]
                best_thres = thres

        if abs(max_gain - original_gini) <= self.min_gain or max_gain == original_gini:
            return LNode(get_majority(data[:, -1]))
        child = {}
        for v, s in best_split.items():
            child[v] = self.build_tree_gini(np.array(s), attributes, depth + 1)
        return DNode(best_attr, child, data, is_categorical, best_thres)

    def predict_one(self, entry):
        return self.root.predict(entry)

    def predict(self, data):
        return np.array([self.root.predict(e) for e in data])


class RandomForestClassifier:
    def __init__(self, ntree=3, min_size_split=2, min_gain=0.0, max_depth=None):
        self.ntree = ntree
        self.min_size_split = min_size_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.trees = []

    def fit_tree(self, X, y, attr_type, criterion):
        tree = DecisionTreeClassifier(
            min_size_split=self.min_size_split,
            max_depth=self.max_depth,
            min_gain=self.min_gain,
        )
        X_bstrap, y_bstrap = bootstrap(X, y)
        tree.fit(X_bstrap, y_bstrap, attr_type, criterion)
        return tree

    def fit(self, X, y, attr_type, criterion="ig"):
        # Use concurrent to construct tree in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(self.ntree):
                futures.append(
                    executor.submit(self.fit_tree, X, y, attr_type, criterion)
                )

            self.trees = [future.result() for future in futures]

    def predict_one(self, X):
        result = np.empty(self.ntree)

        # Calculate the majority
        for i in range(self.ntree):
            result[i] = self.trees[i].predict_one(X)

        return get_majority(result)

    def predict(self, X):
        return np.array([self.predict_one(e) for e in X])
