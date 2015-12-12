# File: algorithm.py
# --------------------
# Extracts features for each face from the face_dict.txt file, then
# runs stochastic gradient descent to train the predictor.

import copy
import random
import collections
import math
import sys
import ast
from util import *
from collections import Counter


def extract_features(attributes):
    """
    Extract features for a face corresponding to the face_id and with
    attributes in the dict face_attrs.
    """

    feature_vector = collections.defaultdict(lambda: 0)

    # top of eyebrows
    brow_line = (attributes['left_eyebrow_upper_middle']['y'] + attributes['right_eyebrow_upper_middle']['y']) / 2
    # top of nose
    nose_line = (attributes['nose_contour_left1']['y'] + attributes['nose_contour_right1']['y']) / 2
    # middle of eyes
    eye_line = (attributes['left_eye_center']['y'] + attributes['right_eye_center']['y']) / 2
    
    # length of face from brow line to chin
    len_face = brow_line - attributes['contour_chin']['y']
    feature_vector['len_face'] = len_face

    # width of face at cheekbones
    width_face_cheekbones = attributes['contour_right2']['x'] - attributes['contour_left2']['x']
    feature_vector['width_face_cheekbones'] = width_face_cheekbones

    # width of face at mouth
    width_face_mouth = attributes['contour_right5']['x'] - attributes['contour_left5']['x']
    feature_vector['width_face_mouth'] = width_face_mouth

    # width of face at eyes
    width_face_eyes = attributes['contour_right1']['x'] - attributes['contour_left1']['x']
    feature_vector['width_face_eyes'] = width_face_eyes

    # distance from left cheek to middle of mouth
    left_cheek_to_mouth = attributes['contour_left5']['x'] - attributes['mouth_upper_lip_bottom']['x']
    # feature_vector['left_cheek_to_mouth'] = left_cheek_to_mouth

    # distance from right cheek to middle of mouth
    right_cheek_to_mouth = attributes['contour_right5']['x'] - attributes['mouth_upper_lip_bottom']['x']
    # feature_vector['right_cheek_to_mouth'] = right_cheek_to_mouth

    # average distance from cheek to middle of mouth
    cheek_to_mouth = (left_cheek_to_mouth + right_cheek_to_mouth) / 2
    feature_vector['cheek_to_mouth'] = cheek_to_mouth

    # width of mouth
    width_mouth = attributes['mouth_right_corner']['x'] - attributes['mouth_left_corner']['x']
    # feature_vector['width_mouth'] = width_mouth

    # thickness of mouth
    thick_mouth = attributes['mouth_upper_lip_top']['y'] - attributes['mouth_lower_lip_bottom']['y']
    feature_vector['thick_mouth'] = thick_mouth

    # height of left eye
    height_left_eye = attributes['left_eye_top']['y'] - attributes['left_eye_bottom']['y']
    # feature_vector['height_left_eye'] = height_left_eye

    # height of right eye
    height_right_eye = attributes['right_eye_top']['y'] - attributes['right_eye_bottom']['y']
    # feature_vector['height_right_eye'] = height_right_eye

    # average height of eye
    height_eye = (height_left_eye + height_right_eye) / 2
    feature_vector['height_eye'] = height_eye

    # width of left eye
    width_left_eye = attributes['left_eye_right_corner']['x'] - attributes['left_eye_left_corner']['x']
    # feature_vector['width_left_eye'] = width_left_eye

    # width of right eye
    width_right_eye = attributes['right_eye_right_corner']['x'] - attributes['right_eye_left_corner']['x']
    # feature_vector['width_right_eye'] = width_right_eye

    # average width of eye
    width_eye = (width_left_eye + width_right_eye) / 2
    feature_vector['width_eye'] = width_eye

    # outer width of eyes
    outer_width_eye = attributes['right_eye_right_corner']['x'] - attributes['left_eye_left_corner']['x']
    # feature_vector['outer_width_eye'] = outer_width_eye

    # distance between inner corner of eyes
    eye_sep = attributes['right_eye_left_corner']['x'] - attributes['left_eye_right_corner']['x']
    feature_vector['eye_sep'] = eye_sep

    # height of left eyebrow
    height_left_eyebrow = attributes['left_eyebrow_upper_middle']['y'] - attributes['left_eyebrow_lower_middle']['y']
    feature_vector['height_left_eyebrow'] = height_left_eyebrow

    # height of right eyebrow
    height_right_eyebrow = attributes['right_eyebrow_upper_middle']['y'] - attributes['right_eyebrow_lower_middle']['y']
    feature_vector['height_right_eyebrow'] = height_right_eyebrow

    # average height of eyebrow
    height_eyebrow = (height_left_eyebrow + height_right_eyebrow) / 2
    feature_vector['height_eyebrow'] = height_eyebrow

    # width of left eyebrow
    width_left_eyebrow = attributes['left_eyebrow_right_corner']['x'] - attributes['left_eyebrow_left_corner']['x']
    feature_vector['width_left_eyebrow'] = width_left_eyebrow

    # width of right eyebrow
    width_right_eyebrow = attributes['right_eyebrow_right_corner']['x'] - attributes['right_eyebrow_left_corner']['x']
    feature_vector['width_right_eyebrow'] = width_right_eyebrow

    # average width of eyebrow
    width_eyebrow = (width_left_eyebrow + width_right_eyebrow) / 2
    # feature_vector['width_eyebrow'] = width_eyebrow

    # length of nose
    len_nose = nose_line - attributes['nose_contour_lower_middle']['y']
    feature_vector['len_nose'] = len_nose

    # width of nose
    width_nose = attributes['nose_right']['x'] - attributes['nose_left']['x']
    feature_vector['width_nose'] = width_nose

    # width of nose at eyes
    width_nose_eyes = attributes['nose_contour_right1']['x'] - attributes['nose_contour_left1']['x']
    feature_vector['width_nose_eyes'] = width_nose_eyes

    # height of left ear
    height_left_ear = attributes['contour_left1']['y'] - attributes['contour_left3']['y']
    feature_vector['height_left_ear'] = height_left_ear

    # height of right ear
    height_right_ear = attributes['contour_right1']['y'] - attributes['contour_right3']['y']
    feature_vector['height_right_ear'] = height_right_ear

    # average height of ear
    height_ear = (height_left_ear + height_right_ear) / 2
    feature_vector['height_ear'] = height_ear

    # length of chin
    len_chin = attributes['mouth_lower_lip_bottom']['y'] - attributes['contour_chin']['y']
    feature_vector['len_chin'] = len_chin

    # width of chin
    width_chin = attributes['contour_right7']['x'] - attributes['contour_left7']['x']
    feature_vector['width_chin'] = width_chin

    # eye area
    eye_area = width_eye * height_eye
    feature_vector['eye_area'] = eye_area

    # face area (dimensions: length of face and width of face at cheekbones)
    face_area = len_face * width_face_cheekbones
    feature_vector['face_area'] = face_area

    # chin area
    chin_area = len_chin * width_chin
    # feature_vector['chin_area'] = chin_area

    # eye height to length of face ratio
    height_eye_len_face = height_eye / len_face
    # feature_vector['height_eye_len_face'] = height_eye_len_face

    # eye width to width of face at eyes ratio
    width_eye_width_face = width_eye / width_face_eyes
    # feature_vector['width_eye_width_face'] = width_eye_width_face

    # eye separation to width of face at eyes ratio
    eye_sep_width_face = eye_sep / width_face_eyes
    # feature_vector['eye_sep_width_face'] = eye_sep_width_face

    # eye area to face area ratio
    eye_area_face_area = eye_area / face_area
    # feature_vector['eye_area_face_area'] = eye_area_face_area

    # outer eye width to width of face at eyes ratio
    outer_width_eye_width_face = outer_width_eye / width_face_cheekbones
    # feature_vector['outer_width_eye_width_face'] = outer_width_eye_width_face

    # nose length to length of face ratio
    len_nose_len_face = len_nose / len_face
    # feature_vector['len_nose_len_face'] = len_nose_len_face

    # nose width to width of face at cheekbones ratio
    width_nose_width_face = width_nose / width_face_cheekbones
    # feature_vector['width_nose_width_face'] = width_nose_width_face

    # nostril flare
    nostril_flare = width_nose / width_nose_eyes
    # feature_vector['nostril_flare'] = nostril_flare

    # width of nose to width of mouth ratio
    width_nose_width_mouth = width_nose / width_mouth
    # feature_vector['width_nose_width_mouth'] = width_nose_width_mouth

    # length of chin to length of face ratio
    len_chin_len_face = len_chin / len_face
    # feature_vector['len_chin_len_face'] = len_chin_len_face

    # width of chin to width of face at cheekbones ratio
    width_chin_width_face = width_chin / width_face_cheekbones
    # feature_vector['width_chin_width_face'] = width_chin_width_face

    # chin area to face area ratio
    chin_area_face_area = chin_area / face_area
    # feature_vector['chin_area_face_area'] = chin_area_face_area

    # cheekbone prominence
    cheek_prom = abs(width_face_cheekbones - width_face_mouth) / len_face
    # feature_vector['cheek_prom'] = cheek_prom

    # facial narrowness (length of face to width of face at mouth ratio)
    facial_narrow = len_face / width_face_mouth
    # feature_vector['facial_narrow'] = facial_narrow

    # eyebrow height to length of face ratio
    height_eyebrow_len_face = height_eyebrow / len_face
    # feature_vector['height_eyebrow_len_face'] = height_eyebrow_len_face

    # eyebrow width to width of face at cheekbones
    width_eyebrow_width_face = width_eyebrow / width_face_cheekbones
    # feature_vector['width_eyebrow_width_face'] = width_eyebrow_width_face

    # eye width to eyebrow width ratio
    width_eye_width_eyebrow = width_eye / width_eyebrow
    # feature_vector['width_eye_width_eyebrow'] = width_eye_width_eyebrow

    # ear height to length of face ratio
    height_ear_len_face = height_ear / len_face
    # feature_vector['height_ear_len_face'] = height_ear_len_face

    return feature_vector


def learn_predictor(train, dev, feature_extractor, evaluate):
    train_data = read_file(train)

    weights = collections.defaultdict(lambda: 0)  # feature => weight
    num_iters = 500

    for t in range(num_iters):
        eta = .00001 # .00001
        for person_id, attrs in train_data.items():
            feature_vector = extract_features(attrs['attributes'])
            rating = float(attrs['rating'])
            margin = dot_product(weights, feature_vector) - rating
            increment(weights, -eta * 2 * margin, feature_vector)

        # train_correct = evaluate_predictor(train_data, extract_features, weights)
        # dev_correct = evaluate_predictor(dev_data, extract_features, weights)
        # print "Official: train = %s, dev = %s" % (train_correct, dev_correct)

    if evaluate:
        dev_data = read_file(dev)

        correct = 0
        results_file = open('dev_results.txt', 'w')
        for person_id, attrs in dev_data.items():
            feature_vector = extract_features(attrs['attributes'])
            rating = float(attrs['rating'])
            score = dot_product(weights, feature_vector)
            if abs(rating - score) <= 0.5: correct += 1
            results_file.write(person_id + ' ' + str(score) + '\n')
        results_file.close()
        print 'correct: ' + str(correct)
        print 'percentage: ' + str(float(correct)/199)
        print weights

    return weights


def main(argv):
    learn_predictor(argv[0], argv[1], extract_features, True)

if __name__ == "__main__":
    main(sys.argv[1:])

