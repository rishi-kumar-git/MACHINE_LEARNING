# =========================================
# CNN FEATURE EXTRACTION + MODEL TRAINING
# =========================================

import os
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------------------
# IMAGE TRANSFORM
# -----------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -----------------------------------------
# LOAD MODELS
# -----------------------------------------

def get_model(model_name):

    if model_name == "resnet":
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif model_name == "vgg":
        model = models.vgg16(pretrained=True)
        model = model.features
        model = torch.nn.Sequential(model, torch.nn.AdaptiveAvgPool2d((1,1)))

    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model = model.features
        model = torch.nn.Sequential(model, torch.nn.AdaptiveAvgPool2d((1,1)))

    model.eval()
    return model


# -----------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------

def extract_features(image_path, model):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.flatten().numpy()


# -----------------------------------------
# BUILD DATASET
# -----------------------------------------

def build_dataset(dataset_path, model):

    X = []
    y = []

    classes = sorted(os.listdir(dataset_path))

    for label, class_name in enumerate(classes):

        class_path = os.path.join(dataset_path, class_name)

        for img_name in os.listdir(class_path):

            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(class_path, img_name)

            feat = extract_features(img_path, model)

            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)


# -----------------------------------------
# EVALUATION FUNCTION
# -----------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "Train Acc": accuracy_score(y_train, train_pred),
        "Test Acc": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, average="macro"),
        "Recall": recall_score(y_test, test_pred, average="macro"),
        "F1": f1_score(y_test, test_pred, average="macro"),
    }


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    dataset_path = "Lab2_Dataset"

    # Try different CNN models
    for model_name in ["resnet", "vgg", "alexnet"]:

        print(f"\n🔷 Using {model_name.upper()}")

        cnn_model = get_model(model_name)

        X, y = build_dataset(dataset_path, cnn_model)

        print("Feature shape:", X.shape)

        # Save features
        np.save(f"X_{model_name}.npy", X)
        np.save(f"y_{model_name}.npy", y)

        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Models
        models_dict = {
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "MLP": MLPClassifier(max_iter=500)
        }

        # Train + Evaluate
        for name, clf in models_dict.items():

            clf.fit(X_train, y_train)

            results = evaluate_model(clf, X_train, X_test, y_train, y_test)

            print(f"\n{name} Results:")
            for k, v in results.items():
                print(f"{k}: {v:.4f}")