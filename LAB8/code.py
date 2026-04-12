# =========================================
# LAB 08 – COMPLETE CODE (A1–A12) FINAL
# =========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# ----------------------------
# BASIC MODULES
# ----------------------------

def summation(x, w):
    return np.dot(x, w)

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def error(y_true, y_pred):
    return y_true - y_pred


# ----------------------------
# PERCEPTRON TRAINING
# ----------------------------

def train_perceptron(X, y, w, lr, activation, max_epochs=1000):
    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1)
            net = summation(x_i, w)
            y_pred = activation(net)

            e = error(y[i], y_pred)
            w = w + lr * e * x_i

            total_error += e**2

        errors.append(total_error)

        if total_error <= 0.002:
            break

    return w, errors, epoch + 1


# ----------------------------
# DATA
# ----------------------------

X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])


# ----------------------------
# A2 – AND
# ----------------------------

w_init = np.array([10, 0.2, -0.75])
lr = 0.05

w_and, err_and, ep_and = train_perceptron(X_and, y_and, w_init.copy(), lr, step)


# ----------------------------
# A3 – ACTIVATIONS
# ----------------------------

activations = {
    "Bipolar": bipolar_step,
    "Sigmoid": lambda x: 1 if sigmoid(x)>=0.5 else 0,
    "ReLU": lambda x: 1 if relu(x)>0 else 0
}

activation_epochs = {}

for name, func in activations.items():
    _, _, ep = train_perceptron(X_and, y_and, w_init.copy(), lr, func)
    activation_epochs[name] = ep


# ----------------------------
# A4 – LEARNING RATE
# ----------------------------

learning_rates = np.arange(0.1, 1.1, 0.1)
lr_epochs = []

for lr_val in learning_rates:
    _, _, ep = train_perceptron(X_and, y_and, w_init.copy(), lr_val, step)
    lr_epochs.append(ep)


# ----------------------------
# A5 – XOR
# ----------------------------

w_xor, err_xor, ep_xor = train_perceptron(X_xor, y_xor, w_init.copy(), lr, step)


# ----------------------------
# A6 – CUSTOMER DATA
# ----------------------------

X_cust = np.array([
[20,6,2,386],
[16,3,6,289],
[27,6,2,393],
[19,1,2,110],
[24,4,2,280],
[22,1,5,167],
[15,4,2,271],
[18,4,2,274],
[21,1,4,148],
[16,2,4,198]
])

y_cust = np.array([1,1,1,0,1,0,1,1,0,0])

X_cust = (X_cust - X_cust.mean(axis=0)) / X_cust.std(axis=0)

w_init2 = np.random.rand(5)

w_cust, err_cust, ep_cust = train_perceptron(
    X_cust, y_cust, w_init2, 0.05,
    lambda x: 1 if sigmoid(x)>=0.5 else 0
)


# ----------------------------
# A7 – PSEUDO INVERSE
# ----------------------------

X_bias = np.c_[np.ones(X_cust.shape[0]), X_cust]
w_pseudo = np.linalg.pinv(X_bias).dot(y_cust)


# ----------------------------
# A8 – BACKPROP
# ----------------------------

def sigmoid_deriv(x):
    return x * (1 - x)

def train_nn(X, y, lr=0.05, epochs=1000):

    np.random.seed(1)

    w1 = np.random.rand(2,2)
    w2 = np.random.rand(2,1)

    errors = []

    for _ in range(epochs):
        hidden = sigmoid(np.dot(X, w1))
        output = sigmoid(np.dot(hidden, w2))

        err = y.reshape(-1,1) - output

        d_output = err * sigmoid_deriv(output)
        d_hidden = d_output.dot(w2.T) * sigmoid_deriv(hidden)

        w2 += hidden.T.dot(d_output) * lr
        w1 += X.T.dot(d_hidden) * lr

        errors.append(np.mean(err**2))

    return w1, w2, errors


# ----------------------------
# A9 – XOR NN
# ----------------------------

w1_xor, w2_xor, err_nn_xor = train_nn(X_xor, y_xor)


# ----------------------------
# A11 – MLP (AND & XOR)
# ----------------------------

mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp_and.fit(X_and, y_and)

mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)
mlp_xor.fit(X_xor, y_xor)


# ----------------------------
# A12 – YOUR PROJECT DATA
# ----------------------------

X = np.load("X_resnet.npy")
y = np.load("y_resnet.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlp_proj = MLPClassifier(max_iter=500)
mlp_proj.fit(X_train, y_train)

acc_proj = mlp_proj.score(X_test, y_test)


# ----------------------------
# OUTPUT
# ----------------------------

print("A2 AND Epochs:", ep_and)
print("A3 Activation Epochs:", activation_epochs)
print("A4 LR vs Epochs:", list(zip(learning_rates, lr_epochs)))
print("A5 XOR Epochs:", ep_xor)
print("A6 Customer Epochs:", ep_cust)
print("A7 Pseudo Weights:", w_pseudo)
print("A9 XOR NN Error:", err_nn_xor[-1])
print("A11 AND Accuracy:", mlp_and.score(X_and, y_and))
print("A11 XOR Accuracy:", mlp_xor.score(X_xor, y_xor))
print("A12 PROJECT Accuracy:", acc_proj)
