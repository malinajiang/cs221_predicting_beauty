# File: util.py 
# --------------------
# Basic helper functions.

import pickle
import os
import random
import operator
import sys
from collections import Counter


def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

def dot_product(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dot_product(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def norm(x1, x2, y1, y2):
    return math.sqrt((abs(x2 - x1))**2 + (abs(y2 - y1))**2)


def slope(x1, x2, y1, y2):
    return (y2 - y1) / (x2 - x1)


def tan_theta(m1, m2):
    return (m1 - m2) / (1 + (m1 * m2))


def read_file(file_name):
    data_file = open(file_name, 'rb')

    # dictionary of person_id : {'url', 'rating', 'face_id', 'attributes}
    data = pickle.load(data_file)
    data_file.close()

    return data