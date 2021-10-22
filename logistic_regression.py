import numpy as np
from numpy import random
from math import e,log

def h(w_values, x_features, b):
    a = np.dot(w_values, x_features) + b
    return a

def s(w_values, x_features, b):
    return 1 / ( 1 + e** (-h(w_values,x_features,b)) )

def error(w_values, x_features, y_values, b):
    err = 0
    n = len(y_values)

    for i in range(n):
        x_temp = np.array(x_features.loc[i])
        s_value = s(w_values,x_temp,b)
        if s_value>0 and 1-s_value>0:
            err += y_values[i]*log(s_value) + (1-y_values[i])*log(1-s_value);

    return (-1/n)*(err)

def derivates(y, w, x, b,k):
    n = len(y)
    dw = [0] * len(w)
    db = b
    for i in range(k):
        for j in range(n):
            x_temp = np.array(x.loc[j])
            #dw[i] += (e ** h(w, x_temp, b) * w[i]) / (1 + e ** (h(w, x_temp, b)))
            dw[i] += (h(w, x_temp, b) - y[i])*x_temp[i]
        #db += (e ** h(w, x_temp, b)) / (1 + e ** (h(w, x_temp, b)))
        dw[i] /= n
        db += h(w, x_temp, b) - y[i]
    return dw, db/n

def update(w_values,b,dw_values,db,alpha,k):
    w_new = [w_values[j] - alpha*dw_values[j] for j in range(k)]
    b_new = b - alpha*db
    return w_new, b_new

def regression(x_train, y_train, x_valid, y_valid, alpha, epochs, k):
    w = list(random.rand(k))
    b = random.rand()
    train_list = []
    valid_list = []
    for i in range(epochs):
        dw, db = derivates(y_train, w, x_train, b,k)
        w,b = update(w,b,dw,db,alpha,k)
        temp_train = error(w, x_train, y_train, b)
        temp_valid = error(w, x_valid, y_valid, b)
        train_list.append(temp_train)
        valid_list.append(temp_valid)
    return train_list, valid_list