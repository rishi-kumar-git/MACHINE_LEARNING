# =========================================
# LAB 09 – STACKING + PIPELINE + LIME (FINAL FIXED)
# =========================================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lime.lime_tabular import LimeTabularExplainer


# =========================================
# LOAD DATA
# =========================================

def load_data():
    X = np.load("X_resnet.npy")
    y = np.load("y_resnet.npy")
    return train_test_split(X, y, test_size=0.3, random_state=42)


# =========================================
# STACKING MODEL
# =========================================

def build_stacking_model():

    base_models = [
        ("svm", SVC(probability=True)),
        ("rf", RandomForestClassifier()),
        ("mlp", MLPClassifier(max_iter=500))
    ]

    meta_model = LogisticRegression(max_iter=500)

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3 
    )

    return stacking


# =========================================
# PIPELINE
# =========================================

def build_pipeline():

    stacking_model = build_stacking_model()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", stacking_model)
    ])

    return pipeline


# =========================================
# EVALUATION
# =========================================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro")
    }


# =========================================
# LIME EXPLANATION (FIXED)
# =========================================

def explain_with_lime(model, X_train, X_test):

    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    class_names = [str(c) for c in np.unique(model.named_steps["classifier"].classes_)]

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba,
        num_features=10
    )

    return exp


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data()

    pipeline = build_pipeline()

    results = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    print("\nSTACKING + PIPELINE RESULTS:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    explanation = explain_with_lime(pipeline, X_train, X_test)

    print("\nLIME Explanation:")
    for item in explanation.as_list():
        print(item)