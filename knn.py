import numpy as np
from numpy.core.numeric import indices
from sklearn.neighbors import KDTree

def calculate_errors(indeces, train, point):
    count_men = 0
    count_women = 0
    for data in indeces:
        if train[data][-1] == 1:
            count_women += 1
        else:
            count_men += 1
    
    if count_men >= count_women and point[-1] == 0:
        return True
    elif count_women >= count_women and point[-1] == 1:
        return True
        
    return False

def knn(train, valid, neighbors):
    tree = KDTree(train[:, :-1], leaf_size = 100)
    cont_errors = 0
    for data in valid:
        distance, indeces = tree.query(data[:-1].reshape(1, -1), neighbors)
        errors = calculate_errors(indeces[0], train, data)
        if errors == False:
            cont_errors += 1
    return cont_errors