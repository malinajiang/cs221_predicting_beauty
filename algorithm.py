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
import ast
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


def extract_features(face_id, face_attrs):
    """
    Extract features for a face corresponding to the face_id and with
    attributes in the dict face_attrs.
    """

    feature_vector = collections.defaultdict(lambda: 0)

    lengthOfFace = abs(face_attrs['attributes']['left_eyebrow_upper_middle']['y'] - face_attrs['attributes']['contour_chin']['y'])
    feature_vector['lenFace'] = lengthOfFace
    widthOfFaceCheekbones = abs(face_attrs['attributes']['contour_right2']['x'] - face_attrs['attributes']['contour_left2']['x'])
    feature_vector['widthCheekbones'] = widthOfFaceCheekbones
    widthOfFaceMouth = abs(face_attrs['attributes']['contour_right5']['x'] - face_attrs['attributes']['contour_left5']['x'])
    feature_vector['widthMouth'] = widthOfFaceMouth
    cheekToMouthDistance = abs(face_attrs['attributes']['contour_right5']['x'] - face_attrs['attributes']['mouth_upper_lip_bottom']['x'])
    feature_vector['cheek2Mouth'] = cheekToMouthDistance
    eyeHeight = abs(face_attrs['attributes']['right_eye_top']['y'] - face_attrs['attributes']['right_eye_bottom']['y']) / lengthOfFace
    feature_vector['iHeight'] = eyeHeight
    eyeWidth = (abs(face_attrs['attributes']['right_eye_right_corner']['x'] - face_attrs['attributes']['right_eye_left_corner']['x'])) / widthOfFaceCheekbones
    feature_vector['iWidth'] = eyeWidth
    eyeArea = eyeHeight * eyeWidth
    feature_vector['iArea'] = eyeArea
    widthOfFaceEye = abs(face_attrs['attributes']['contour_right1']['x'] - face_attrs['attributes']['contour_left1']['x'])
    feature_vector['widthFaceEye'] = widthOfFaceEye
    outerEyeWidth = abs(face_attrs['attributes']['right_eye_right_corner']['x'] - face_attrs['attributes']['left_eye_left_corner']['x']) / widthOfFaceEye
    feature_vector['outerIWidth'] = outerEyeWidth
    innerEyeWidth = abs(face_attrs['attributes']['right_eye_left_corner']['x'] - face_attrs['attributes']['left_eye_right_corner']['x']) / widthOfFaceEye
    feature_vector['innerIWidth'] = innerEyeWidth
    noseLength = abs((abs(face_attrs['attributes']['nose_left']['y'] + face_attrs['attributes']['nose_right']['y']) / 2) - face_attrs['attributes']['nose_contour_lower_middle']['y']) / lengthOfFace
    feature_vector['noseLen'] = noseLength
    noseTipWidth = abs(face_attrs['attributes']['nose_left']['x'] + face_attrs['attributes']['nose_right']['x']) / widthOfFaceMouth
    feature_vector['noseTW'] = noseTipWidth
    noseArea = (noseLength*noseTipWidth) / widthOfFaceMouth
    feature_vector['noseA'] = noseArea
    nostrilWidth = abs(face_attrs['attributes']['nose_contour_left3']['x'] - face_attrs['attributes']['nose_contour_right3']['x']) / widthOfFaceMouth
    feature_vector['nosWid'] = nostrilWidth
    chinLength = abs(face_attrs['attributes']['mouth_upper_lip_bottom']['y'] - face_attrs['attributes']['contour_chin']['y']) / lengthOfFace
    feature_vector['chinLen'] = chinLength
    chinWidth = abs(face_attrs['attributes']['contour_left7']['x'] - face_attrs['attributes']['contour_right7']['x']) / lengthOfFace
    feature_vector['chinWid'] = chinWidth
    chinArea = chinLength * chinWidth
    feature_vector['chinA'] = chinArea
    horizontalEyeSeparation = abs(face_attrs['attributes']['left_eye_pupil']['x'] - face_attrs['attributes']['right_eye_pupil']['x']) / widthOfFaceCheekbones
    feature_vector['horizontalISep'] = horizontalEyeSeparation
    cheekboneProminence = abs(widthOfFaceCheekbones - widthOfFaceMouth) / lengthOfFace
    feature_vector['cheekProm'] = cheekboneProminence
    cheekThinness = (abs(face_attrs['attributes']['mouth_right_corner']['x'] - face_attrs['attributes']['contour_right5']['x'])) / lengthOfFace
    feature_vector['cheekThin'] = cheekThinness
    facialNarrowness = lengthOfFace / widthOfFaceMouth
    feature_vector['facialNarrow'] = facialNarrowness
    eyebrowHeight = abs(face_attrs['attributes']['right_eye_pupil']['y'] - face_attrs['attributes']['right_eyebrow_lower_middle']['y']) / lengthOfFace
    feature_vector['eyebrowHi'] = eyebrowHeight
    upperLipThickness = abs(face_attrs['attributes']['mouth_upper_lip_bottom']['y'] - face_attrs['attributes']['mouth_upper_lip_top']['y']) / lengthOfFace
    feature_vector['upLipThick'] = upperLipThickness
    lowerLipThickness = abs(face_attrs['attributes']['mouth_upper_lip_bottom']['y'] - face_attrs['attributes']['mouth_lower_lip_bottom']['y']) / lengthOfFace
    feature_vector['lowLipThick'] = lowerLipThickness
    lipLength = abs(face_attrs['attributes']['mouth_right_corner']['x'] - face_attrs['attributes']['mouth_left_corner']['x']) / widthOfFaceMouth
    feature_vector['lipLen'] = lipLength
    browThickness = abs(face_attrs['attributes']['right_eyebrow_lower_middle']['y'] - face_attrs['attributes']['right_eyebrow_upper_middle']['y']) / lengthOfFace
    feature_vector['browThick'] = browThickness
    symmetry1 = abs(face_attrs['attributes']['right_eye_pupil']['x'] - face_attrs['attributes']['contour_right1']['x']) / abs(face_attrs['attributes']['contour_left1']['x'] - face_attrs['attributes']['left_eye_pupil']['x'])
    feature_vector['symm1'] = symmetry1
    symmetry2 = abs(face_attrs['attributes']['nose_tip']['x'] - face_attrs['attributes']['contour_right3']['x']) / abs(face_attrs['attributes']['contour_left3']['x'] - face_attrs['attributes']['nose_tip']['x'])
    feature_vector['symm2'] = symmetry2
    symmetry3 = abs(face_attrs['attributes']['mouth_upper_lip_bottom']['x'] - face_attrs['attributes']['contour_right5']['x']) / abs(face_attrs['attributes']['contour_left5']['x'] - face_attrs['attributes']['mouth_upper_lip_bottom']['x'])
    feature_vector['symm3'] = symmetry3
    pupilWidth = abs(face_attrs['attributes']['left_eye_pupil']['x'] - face_attrs['attributes']['right_eye_pupil']['x'])
    feature_vector['pupilWid'] = pupilWidth
    noseVsPupilWidth = abs(face_attrs['attributes']['nose_left']['x'] - face_attrs['attributes']['nose_right']['x']) / pupilWidth
    feature_vector['nVPW'] = noseVsPupilWidth

    # lipVsPupil = lipLength/pupilWidth
    # noseVsLip = noseTipWidth/lipLength
    # angleCheekboneLowerJaw = 
    # anglePupilLowerJaw1 = 
    # anglePupilLowerJaw2 = 
    # pupilVsNoseWidth = abs(face_attrs['attributes']['nose_left']['x'] - face_attrs['attributes']['nose_right']['x'])/ ######
    # pupilVsLowerJaw = 
    # pupilVsWidthOfFace = 
    # pupilVsPupil = 
    # widthOfFaceVsWidthOfFace = widthOfFaceEye/widthOfFaceMouth
    # lipHeightVsLowerFaceLength = 

    return feature_vector


def learn_predictor(train, dev, feature_extractor):
    train_data = read_file(train)
    dev_data = read_file(dev)

    weights = collections.defaultdict(lambda: 0)  # feature => weight
    num_iters = 20

    for t in range(num_iters):
        # print t
        eta = 1 / math.sqrt(t + 1)   

        for person_id, attrs in train_data.items():

            feature_vector = extract_features(person_id, attrs)
            # print feature_vector
            rating = float(attrs['rating'])
            margin = dot_product(weights, feature_vector) - rating
            # print dot_product(weights, feature_vector)
            # print weights
            # print 'hi'
            # increment(weights, -eta * 2 * margin, feature_vector)
            print weights
            print feature_vector

            increment(weights, -eta * 2 * margin, feature_vector)
            # print margin
            # print weights
            break

        # trainError = evaluatePredictor(train, lambda (x) : (1 if dot_product(extract_features(x), weights) >= 0 else -1))
        # devError = evaluatePredictor(dev, lambda (x) : (1 if dot_product(extract_features(x), weights) >= 0 else -1))
        # print "Official: train error = %s, dev error = %s" % (trainError, devError)


    # results_file = open('dev_results.txt', 'w')
    # for person_id, attrs in dev_data.items():
    #     feature_vector = extract_features(person_id, attrs)
    #     rating = float(attrs['rating'])
    #     score = dot_product(weights, feature_vector)
    #     results_file.write(person_id + ' ' + str(score))
    # results_file.close()

    print weights
    return weights


def main(argv):
    learn_predictor(argv[0], argv[1], extract_features)

if __name__ == "__main__":
    main(sys.argv[1:])






