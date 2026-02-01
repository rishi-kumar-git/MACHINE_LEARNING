import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def dot_product(a, b):
    return np.sum(a * b)


def euclidean_norm(a):
    return np.sqrt(np.sum(a ** 2))


def load_dataset(path):
    X = []
    y = []
    classes = sorted(os.listdir(path))

    for label, folder in enumerate(classes):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, file), 0)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                X.append(img.flatten())
                y.append(label)

    return np.array(X), np.array(y), classes


def minkowski_distance(a, b, p):
    return np.power(np.sum(np.abs(a - b) ** p), 1 / p)


def knn_predict(X_train, y_train, test, k):
    dist = []
    for i in range(len(X_train)):
        d = euclidean_norm(X_train[i] - test)
        dist.append((d, y_train[i]))
    dist.sort()
    labels = [label for _, label in dist[:k]]
    return Counter(labels).most_common(1)[0][0]


def knn_custom(X_train, y_train, X_test, k):
    preds = []
    for sample in X_test:
        preds.append(knn_predict(X_train, y_train, sample, k))
    return np.array(preds)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def main():
    dataset_path = "Lab2_Dataset"   

    X, y, classes = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    a = X[0]
    b = X[1]
    print("Dot Product:", dot_product(a, b))
    print("Vector Length:", euclidean_norm(a))

    p_vals = []
    dist_vals = []
    for p in range(1, 11):
        p_vals.append(p)
        dist_vals.append(minkowski_distance(a, b, p))

    plt.plot(p_vals, dist_vals)
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.show()

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    print("Sklearn kNN Accuracy:", model.score(X_test, y_test))

    y_pred_custom = knn_custom(X_train, y_train, X_test, 3)
    print("Custom kNN Accuracy:", accuracy(y_test, y_pred_custom))

    acc_list = []
    k_vals = range(1, 12)
    for k in k_vals:
        preds = knn_custom(X_train, y_train, X_test, k)
        acc_list.append(accuracy(y_test, preds))

    plt.plot(k_vals, acc_list)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()

    cm = confusion_matrix(y_test, y_pred_custom)
    print("Confusion Matrix:\n", cm)


main()
