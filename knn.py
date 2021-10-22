import numpy as np
from sklearn.neighbors import KDTree

#def calculate_errors():


def KNN(train, valid, neighbors):
    tree = KDTree(train, leaf_size = 100)
    distance, indeces = tree.query()
    #calculate_erros()