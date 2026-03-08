# LAB 06 – DECISION TREE ANALYSIS

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler


# ENTROPY CALCULATION
def entropy_calc(labels):

    unique_vals, freq = np.unique(labels, return_counts=True)
    probs = freq / len(labels)

    ent = 0
    for p in probs:
        ent -= p * np.log2(p)

    return ent


# GINI CALCULATION
def gini_calc(labels):

    unique_vals, freq = np.unique(labels, return_counts=True)
    probs = freq / len(labels)

    g = 1 - np.sum(probs ** 2)

    return g


# DISCRETIZE DATA USING EQUAL WIDTH
def bin_values(arr, k=4):

    low = np.min(arr)
    high = np.max(arr)

    step = (high - low) / k

    result = ((arr - low) // step)

    result[result == k] = k - 1

    return result.astype(int)


# APPLY BINNING TO WHOLE DATASET
def preprocess_bins(data, k=4):

    new_data = np.zeros_like(data)

    for col in range(data.shape[1]):
        new_data[:, col] = bin_values(data[:, col], k)

    return new_data


# INFORMATION GAIN
def info_gain(feature_column, labels):

    total_entropy = entropy_calc(labels)

    vals = np.unique(feature_column)

    weighted_ent = 0

    for v in vals:

        subset = labels[feature_column == v]

        weight = len(subset) / len(labels)

        weighted_ent += weight * entropy_calc(subset)

    return total_entropy - weighted_ent


# FIND BEST SPLIT FEATURE
def choose_root_feature(X, y):

    scores = []

    for f in range(X.shape[1]):
        score = info_gain(X[:, f], y)
        scores.append(score)

    best_feature = np.argmax(scores)

    return best_feature


# TREE NODE
class TreeNode:

    def __init__(self, feature=None, split_val=None, left=None, right=None, prediction=None):

        self.feature = feature
        self.split_val = split_val
        self.left = left
        self.right = right
        self.prediction = prediction


# BUILD SIMPLE TREE
def create_tree(X, y, depth=0, max_depth=3):

    if len(np.unique(y)) == 1:
        return TreeNode(prediction=y[0])

    if depth == max_depth:
        values, counts = np.unique(y, return_counts=True)
        return TreeNode(prediction=values[np.argmax(counts)])

    best_feature = choose_root_feature(X, y)

    vals = np.unique(X[:, best_feature])
    split_val = vals[0]

    left_mask = X[:, best_feature] == split_val
    right_mask = X[:, best_feature] != split_val

    left_child = create_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_child = create_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

    return TreeNode(best_feature, split_val, left_child, right_child)


# PREDICTION
def classify(tree, sample):

    if tree.prediction is not None:
        return tree.prediction

    if sample[tree.feature] == tree.split_val:
        return classify(tree.left, sample)

    return classify(tree.right, sample)


# SKLEARN TREE VISUALIZATION
def draw_tree(X, y, feature_list):

    clf = DecisionTreeClassifier(max_depth=3)

    clf.fit(X, y)

    plt.figure(figsize=(12, 6))

    plot_tree(
        clf,
        feature_names=feature_list,
        class_names=True,
        filled=True
    )

    plt.show()

    return clf


# DECISION BOUNDARY (2 FEATURES)
def decision_surface(X, y):

    model = DecisionTreeClassifier()

    model.fit(X, y)

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1

    grid_x, grid_y = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200)
    )

    mesh = np.c_[grid_x.ravel(), grid_y.ravel()]

    pred = model.predict(mesh)

    pred = pred.reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, pred, alpha=0.3)

    plt.scatter(X[:,0], X[:,1], c=y)

    plt.title("Decision Tree Regions")

    plt.show()


# MAIN EXECUTION
if __name__ == "__main__":

    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")

    print("Shape:", X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ENTROPY
    e = entropy_calc(y)
    print("Entropy:", e)

    # GINI
    g = gini_calc(y)
    print("Gini:", g)

    # BIN DATA
    X_disc = preprocess_bins(X_scaled)

    # BEST FEATURE
    root = choose_root_feature(X_disc, y)
    print("Best root feature:", root)

    # BUILD TREE
    tree_model = create_tree(X_disc, y)

    # VISUALIZE
    features = [f"Feature_{i}" for i in range(X.shape[1])]
    draw_tree(X_scaled, y, features)

    # DECISION BOUNDARY
    decision_surface(X_scaled[:, :2], y)