import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def train_single_attribute_lr(X_train, feature_index):
    y_train = X_train[:, feature_index]
    X_used = np.delete(X_train, feature_index, axis=1)
    reg = LinearRegression()
    reg.fit(X_used, y_train)
    return reg

def predict_single_attribute_lr(reg, X, feature_index):
    X_used = np.delete(X, feature_index, axis=1)
    y_pred = reg.predict(X_used)
    y_true = X[:, feature_index]
    return y_true, y_pred

def regression_scores(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def train_multi_attribute_lr(X_train, target_index):
    y_train = X_train[:, target_index]
    X_used = np.delete(X_train, target_index, axis=1)
    reg = LinearRegression()
    reg.fit(X_used, y_train)
    return reg

def predict_multi_attribute_lr(reg, X, target_index):
    X_used = np.delete(X, target_index, axis=1)
    y_pred = reg.predict(X_used)
    y_true = X[:, target_index]
    return y_true, y_pred

def perform_kmeans(X_train, k=2):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X_train)
    return kmeans

def get_cluster_details(kmeans):
    return kmeans.labels_, kmeans.cluster_centers_

def clustering_metrics(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db

def evaluate_multiple_k(X, k_values):
    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_values:
        kmeans = perform_kmeans(X, k)
        labels = kmeans.labels_
        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    return sil_scores, ch_scores, db_scores

def plot_k_analysis(k_values, sil_scores, ch_scores, db_scores):
    plt.figure()
    plt.plot(k_values, sil_scores, marker='o')
    plt.plot(k_values, ch_scores, marker='o')
    plt.plot(k_values, db_scores, marker='o')
    plt.show()

def elbow_method(X, k_range):
    distortions = []

    for k in k_range:
        kmeans = perform_kmeans(X, k)
        distortions.append(kmeans.inertia_)

    return distortions

def plot_elbow(k_range, distortions):
    plt.figure()
    plt.plot(k_range, distortions, marker='o')
    plt.show()