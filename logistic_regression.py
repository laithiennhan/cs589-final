import numpy as np
import sys
from matplotlib import pyplot as plt
from utils import convert_to_float, load_data, normalize, split_folds, split_train_test, eval

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m 

def gradient(X, y_true, y_pred):
    m = X.shape[0]
    return np.dot(X.T, (y_pred - y_true)) / m

class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        self.weights = np.zeros((n_features, n_classes))
        
        y_one_hot = np.eye(n_classes)[y]
        
        for i in range(self.num_iterations):
            scores = np.dot(X, self.weights)
            y_pred = softmax(scores)
            
            loss = cross_entropy(y_one_hot, y_pred)
            self.losses.append(loss)
            
            grad = gradient(X, y_one_hot, y_pred)
            self.weights -= self.learning_rate * grad
            

    def predict(self, X):
        scores = np.dot(X, self.weights)
        y_pred = softmax(scores)
        return np.argmax(y_pred, axis=1)


def main(filename, is_normalize=True):
    X, y = load_data(filename, encode=True)
    y = np.array([int(i) for i in y])
    num_iterations = 2000
    X = normalize(X)
    k = 10 
    folds = split_folds(y, k)

    losses_over_iterations = np.array([0 for i in range(num_iterations)])
    for i in range(k):
        X_train, X_test, y_train, y_test =  split_train_test(X, y, folds, i)
        classifier = MultinomialLogisticRegression(learning_rate=0.01, num_iterations=num_iterations)
        classifier.fit(X_train, y_train)
        losses_over_iterations = losses_over_iterations + np.array(classifier.losses)
        predictions = classifier.predict(X_test)
        res = eval(y_test, predictions)

    
    losses_over_iterations = losses_over_iterations / k


    accuracies_list = []
    f1_list = []
    iteration_list = range(0, 2001, 100)
    for j in iteration_list:
        accuracies = []
        f1_scores = []
        for i in range(k):
            X_train, X_test, y_train, y_test =  split_train_test(X, y, folds, i)
            classifier = MultinomialLogisticRegression(learning_rate=0.005, num_iterations=j)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            res = eval(y_test, predictions)
            accuracies.append(res['accuracy'])
            f1_scores.append(res['f1'])
        accuracies_list.append(np.mean(accuracies))
        f1_list.append(np.mean(f1_scores))
    
    print(max(accuracies_list))
    print(max(f1_list))
    
    
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)  
    plt.plot(range(0, num_iterations), losses_over_iterations)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Cross Entropy Loss over iterations')

    plt.subplot(2, 2, 2)  
    plt.plot(iteration_list, accuracies_list)
    plt.xlabel('Epoch count')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epoch count')

    plt.subplot(2, 2, 3)  
    plt.plot(iteration_list, f1_list)
    plt.xlabel('Epoch count')
    plt.ylabel('F1 Score')
    plt.title('F1 over epoch count')
    plt.tight_layout()
    plt.savefig(f'./figures/logistics_{filename}.png')

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