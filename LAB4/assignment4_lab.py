import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def met(y, p):
    return confusion_matrix(y, p), precision_score(y, p), recall_score(y, p), f1_score(y, p)

def train_pts():
    x = np.random.randint(1, 11, (20, 2))
    y = np.array([0 if i[0] + i[1] < 12 else 1 for i in x])
    return x, y

def test_pts():
    a = np.arange(0, 10, 0.1)
    b = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(a, b)
    return np.c_[xx.ravel(), yy.ravel()]

def knn(x, y, t, k):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(x, y)
    return m.predict(t)

def bestk(x, y):
    g = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 10)}, cv=5)
    g.fit(x, y)
    return g.best_params_, g.best_score_

def main():
    x, y = make_classification(n_samples=150, n_features=4, random_state=1)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3)

    m = KNeighborsClassifier(n_neighbors=5)
    m.fit(xtr, ytr)

    print(met(ytr, m.predict(xtr)))
    print(met(yte, m.predict(xte)))

    x2, y2 = train_pts()
    plt.scatter(x2[y2==0][:,0], x2[y2==0][:,1])
    plt.scatter(x2[y2==1][:,0], x2[y2==1][:,1])
    plt.show()

    t = test_pts()
    for k in [1,3,5]:
        p = knn(x2, y2, t, k)
        plt.scatter(t[:,0], t[:,1], c=p, s=1)
        plt.show()

    print(bestk(x2, y2))

main()
