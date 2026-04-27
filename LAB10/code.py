import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


###########################################################
# PREPROCESSING (ResNet + 5 strips)
###########################################################

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def split_horizontal(img,n=5):
    h,w=img.shape[:2]
    sh=h//n
    return [img[i*sh:(i+1)*sh,:] for i in range(n)]


def extract_feature(path):

    img=np.array(Image.open(path).convert("RGB"))

    strips=split_horizontal(img,5)

    features=[]

    for s in strips:

        s=Image.fromarray(s)
        s=transform(s).unsqueeze(0)

        with torch.no_grad():
            f=model(s)

        features.extend(f.view(-1).numpy())

    return np.array(features)


def build_dataset(folder):

    X=[]
    y=[]
    label=0

    for person in os.listdir(folder):

        ppath=os.path.join(folder,person)

        if not os.path.isdir(ppath):
            continue

        for img in os.listdir(ppath):

            try:
                feat=extract_feature(os.path.join(ppath,img))
                X.append(feat)
                y.append(label)

            except:
                continue

        label+=1

    return np.array(X),np.array(y)


###########################################################
# LOAD DATA
###########################################################

X,y=build_dataset("Dataset")

print("Dataset shape:",X.shape)

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42
)


###########################################################
# MODEL EVALUATION
###########################################################

def run_models(X_train,X_test,y_train,y_test):

    models_dict={
        "SVM":SVC(),
        "Random Forest":RandomForestClassifier(),
        "Decision Tree":DecisionTreeClassifier()
    }

    results={}

    for name,model in models_dict.items():

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        acc=accuracy_score(y_test,pred)

        results[name]=acc

    return results


###########################################################
# A1 CORRELATION HEATMAP
###########################################################

corr=np.corrcoef(X_train[:,:50].T)

plt.figure(figsize=(8,6))
sns.heatmap(corr,cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


###########################################################
# A2 PCA 99%
###########################################################

pca_99=PCA(n_components=0.99)

X_train_99=pca_99.fit_transform(X_train)
X_test_99=pca_99.transform(X_test)

print("\nPCA 99% shape:",X_train_99.shape)

res_99=run_models(
X_train_99,
X_test_99,
y_train,
y_test
)

print("PCA 99% Results:",res_99)


###########################################################
# A3 PCA 95%
###########################################################

pca_95=PCA(n_components=0.95)

X_train_95=pca_95.fit_transform(X_train)
X_test_95=pca_95.transform(X_test)

print("\nPCA 95% shape:",X_train_95.shape)

res_95=run_models(
X_train_95,
X_test_95,
y_train,
y_test
)

print("PCA 95% Results:",res_95)


###########################################################
# A4 SEQUENTIAL FEATURE SELECTION
# FIXED: use PCA features, not raw 2560 features
###########################################################

print("\nRunning Sequential Feature Selection...")

model_sfs=RandomForestClassifier()

sfs=SequentialFeatureSelector(
model_sfs,
n_features_to_select=5,
direction="forward",
cv=2
)

sfs.fit(X_train_95,y_train)

X_train_sfs=sfs.transform(X_train_95)
X_test_sfs=sfs.transform(X_test_95)

print("SFS shape:",X_train_sfs.shape)

res_sfs=run_models(
X_train_sfs,
X_test_sfs,
y_train,
y_test
)

print("SFS Results:",res_sfs)


###########################################################
# A5 SHAP
###########################################################

print("\nRunning SHAP...")

try:

    import shap

    rf=RandomForestClassifier()
    rf.fit(X_train_95,y_train)

    explainer=shap.Explainer(rf)

    shap_values=explainer(X_test_95)

    shap.summary_plot(
        shap_values,
        X_test_95
    )

except:
    print("SHAP not installed or error")


###########################################################
# A5 LIME
###########################################################

print("\nRunning LIME...")

try:

    from lime.lime_tabular import LimeTabularExplainer

    explainer=LimeTabularExplainer(
        X_train_95,
        mode="classification"
    )

    exp=explainer.explain_instance(
        X_test_95[0],
        rf.predict_proba
    )

    print(exp.as_list())

except:
    print("LIME not installed or error")
