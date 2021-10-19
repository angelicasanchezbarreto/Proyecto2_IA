import numpy as np
from numpy import random

def h(w, x, b):
    a = np.dot(w, x) + b
    return a

def error(w, x, y, b, const):
    grad = np.linalg.norm(w,ord=1) / 2
    err = 0
    n = len(x)

    for i in range(n):
        err += max(0, 1 - y.iloc[i] * h(w, x.iloc[i].tolist(), b))

    return grad + const*err

def derivates(y, w, x, b, const, alpha, epochs):
    n = len(w)
    dw = [0] * len(w)
    db = b
    for i in range(n):
        temp = (y * h(w, x, b))
        if temp > 1:
            dw[i] = w[i] - alpha * w[i] * const * (1/epochs)
        else:
            dw[i] = w[i] - alpha * ((1/epochs) * w[i] - const * y * x[i])
            db = b - alpha * (-y * const)

    return dw, db

def regression(x_train, y_train, x_valid, y_valid, alpha, epochs, c, k):
    w = list(random.rand(k))
    b = random.rand()
    train_list = []
    valid_list = []
    for i in range(epochs):
        p = random.randint(len(x_train))
        x_list = x_train.iloc[p].tolist()
        y_list = y_train.iloc[p]
        w, b = derivates(y_list, w, x_list, b, c, alpha, epochs)
        temp_train = error(w, x_train, y_train, b, c)
        temp_valid = error(w, x_valid, y_valid, b, c)
        train_list.append(temp_train)
        valid_list.append(temp_valid)
    return train_list, valid_list,w