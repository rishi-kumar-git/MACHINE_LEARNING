import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from scipy.stats import randint, uniform
import pandas as pd

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except:
    XGBOOST_OK = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except:
    CATBOOST_OK = False


def load_dataset(path="Lab2_Dataset", img_size=(64, 64)):
    X, y = [], []
    for person in sorted(os.listdir(path)):
        folder = os.path.join(path, person)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = imread(os.path.join(folder, fname))
            if img.ndim == 3:
                img = rgb2gray(img)
            img = resize(img, img_size, anti_aliasing=True)
            X.append(img)
            y.append(person)
    return np.array(X), np.array(y)


def get_hog_features(images):
    features = []
    for img in images:
        f = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features.append(f)
    return np.array(features)


def get_lbp_features(images):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, 8, 1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist = hist.astype(float) / (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)


def extract_features(images):
    hog_f = get_hog_features(images)
    lbp_f = get_lbp_features(images)
    return np.hstack([hog_f, lbp_f])


def evaluate(name, model, X_train, X_test, y_train, y_test, cv):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, np.vstack([X_train, X_test]),
                                np.hstack([y_train, y_test]), cv=cv, scoring='accuracy')

    return {
        "Model":       name,
        "Train Acc":   round(accuracy_score(y_train, train_pred) * 100, 2),
        "Test Acc":    round(accuracy_score(y_test, test_pred) * 100, 2),
        "Precision":   round(precision_score(y_test, test_pred, average='macro', zero_division=0) * 100, 2),
        "Recall":      round(recall_score(y_test, test_pred, average='macro', zero_division=0) * 100, 2),
        "F1 Score":    round(f1_score(y_test, test_pred, average='macro', zero_division=0) * 100, 2),
        "CV Acc":      round(cv_scores.mean() * 100, 2),
        "CV Std":      round(cv_scores.std() * 100, 4),
    }


def tune_hyperparams(X_train, y_train):
    svm_search = RandomizedSearchCV(
        SVC(),
        {'C': uniform(0.1, 10), 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']},
        n_iter=15, cv=3, scoring='accuracy', random_state=42, n_jobs=-1
    )
    svm_search.fit(X_train, y_train)
    print(f"SVM best params: {svm_search.best_params_}  score: {svm_search.best_score_*100:.2f}%")

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': randint(50, 200), 'max_depth': [None, 5, 10, 15], 'min_samples_split': randint(2, 10)},
        n_iter=15, cv=3, scoring='accuracy', random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    print(f"RF best params:  {rf_search.best_params_}  score: {rf_search.best_score_*100:.2f}%")

    return svm_search.best_estimator_, rf_search.best_estimator_


def plot_results(df):
    models = df["Model"]
    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(x - w/2, df["Train Acc"], w, label="Train", color='steelblue')
    axes[0].bar(x + w/2, df["Test Acc"],  w, label="Test",  color='salmon')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Train vs Test Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].plot(models, df["Precision"], 'o-', label="Precision", color='green')
    axes[1].plot(models, df["Recall"],    's-', label="Recall",    color='blue')
    axes[1].plot(models, df["F1 Score"],  '^-', label="F1",        color='red')
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Precision / Recall / F1")
    axes[1].legend()
    axes[1].set_ylim(0, 110)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("lab7_results.png", dpi=150)
    plt.show()


def plot_cm(model, X_test, y_test, class_names):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("lab7_confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":

    X_images, y_labels = load_dataset("Lab2_Dataset", img_size=(64, 64))

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    class_names = le.classes_

    X = extract_features(X_images)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Features: {X.shape[1]}")

    # A2 - hyperparameter tuning
    best_svm, best_rf = tune_hyperparams(X_train, y_train)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # A3 - all classifiers
    classifiers = {
        "Perceptron":    Perceptron(max_iter=1000, random_state=42),
        "SVM":           SVC(kernel='rbf', random_state=42),
        "SVM (Tuned)":   best_svm,
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "RF (Tuned)":    best_rf,
        "AdaBoost":      AdaBoostClassifier(n_estimators=50, random_state=42),
        "Naive Bayes":   GaussianNB(),
        "MLP":           MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

    if XGBOOST_OK:
        classifiers["XGBoost"] = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
    if CATBOOST_OK:
        classifiers["CatBoost"] = CatBoostClassifier(iterations=100, verbose=0, random_state=42)

    results = []
    for name, clf in classifiers.items():
        try:
            res = evaluate(name, clf, X_train, X_test, y_train, y_test, cv)
            results.append(res)
            print(f"{name:20s}  Train: {res['Train Acc']}%  Test: {res['Test Acc']}%  F1: {res['F1 Score']}%")
        except Exception as e:
            print(f"{name} failed: {e}")

    # A4 - results table
    df = pd.DataFrame(results).sort_values("Test Acc", ascending=False).reset_index(drop=True)
    df.index += 1
    print("\n", df.to_string())
    df.to_csv("lab7_results.csv", index=False)

    # A5 - plots
    plot_results(df)

    best_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    best_clf.fit(X_train, y_train)
    plot_cm(best_clf, X_test, y_test, class_names)