import numpy as np
from sklearn.neighbors import KDTree

def calculate_errors(indeces, train, point):
    count_men = 0
    count_women = 0
    for data in indeces:
        if train[data][-1] == 1:
            count_women
+= 1
        else:
            count_men += 1
    
    if count_men > count_women && point[-1] == 0:
        return True
    elif count_women > count_men 
        
    return True

def knn(train, valid, neighbors):
    tree = KDTree(train, leaf_size = 100)
    for data in valid:
        distance, indeces = tree.query(data, neighbors)
        errors = calculate_errors(indeces, train, data)
    return errors