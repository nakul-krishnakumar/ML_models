import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    X_train: for each example, contains 3 features:

          - Ear Shape (1 if pointy, 0 otherwise)
          - Face Shape (1 if round, 0 otherwise)
          - Whiskers (1 if present, 0 otherwise)

    y_train: whether the animal is a cat

          - 1 if the animal is a cat
          - 0 otherwise

"""

X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

def entropy(p):
    """
        H(p) -> entropy : measure of impurity in a split  
    """
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p)*np.log2(1 - p)
    
def split_indices(X, index_feature):
    """ 
    General function to split a node into two nodes (left & right), based on the mentioned index_feature.
    Here,
        index feature = 0 => ear shape
        index feature = 1 => face shape
        index feature = 2 => whiskers
    """

    left_indices = []
    right_indices = []

    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    
    return left_indices, right_indices

def weighted_entropy(X, y, left_indices, right_indices):
    """
        This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)

    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

def information_gain(X, y, left_indices, right_indices):
    """
        Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy


# Computing information gain for each feature
for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)

    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
    """
        Select the one with highest information gain
        Here, Ear Shape,
        Similarly we will choose next features recursively
    """