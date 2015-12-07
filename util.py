# File: util.py 
# --------------------
# Basic helper functions.

import pickle
import os
import random
import operator
import sys
from collections import Counter


def evaluate_predictor(data, feature_extractor, weights):
    correct = 0

    for person_id, attrs in data.items():
        feature_vector = feature_extractor(attrs['attributes'])
        rating = float(attrs['rating'])
        prediction = dot_product(weights, feature_vector)
        if abs(rating - prediction) <= 0.5: correct += 1
        
    return correct

def dot_product(d1, d2):
    if len(d1) < len(d2):
        return dot_product(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())


def increment(d1, scale, d2):
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

    