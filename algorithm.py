# File: algorithm.py
# --------------------
# Extracts features for each face from the face_dict.txt file, then
# runs stochastic gradient descent to train the predictor.

import pickle
import copy
import random
import collections
import math
import sys
from collections import Counter
from util import *


def read_file(file_name):
    data_file = open(file_name, 'rb')

    # dictionary of person_id : {'url', 'rating', 'face_id', 'features'}
    data = pickle.load(data_file)
    data_file.close()

    return data

def extract_features(x):
    """
    Extract features for a dict x, which represents the attributes
    dictionary of a face.
    @param dict x: 
    @return dict: feature vector representation of x.
    """

    feature_vector = collections.defaultdict(lambda: 0)
    words = x.split()

    for word in words:
        feature_vector[word] += 1
    
    return feature_vector


def learn_predictor(train, dev, feature_extractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    numIters refers to a variable you need to declare. It is not passed in.
    '''

    train_data = read_file(train)
    dev_data = read_file(dev)

    weights = collections.defaultdict(lambda: 0)  # feature => weight
    num_iters = 20

    for t in range(num_iters):
        eta = 1/math.sqrt(t + 1)   

        for x, y in training_set:
            feature_vector = featureExtractor(x)
            margin = y * dotProduct(weights, feature_vector)
 
            if margin < 1:
                increment(weights, eta * y, feature_vector)

        # trainError = evaluatePredictor(trainExamples, lambda (x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        # devError = evaluatePredictor(testExamples, lambda (x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        # print "Official: train error = %s, dev error = %s" % (trainError, devError)

    return weights


def main(argv):
    learn_predictor(argv[0), argv[1], extract_features)

if __name__ == "__main__":
    main(sys.argv[1:])






