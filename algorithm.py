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

    # 1. length of face
    len_face = abs(attributes['left_eyebrow_upper_middle']['y'] - attributes['contour_chin']['y'])
    feature_vector['len_face'] = len_face
    # 2. width of face at cheekbones
    width_face_cheekbones = abs(attributes['contour_right2']['x'] - attributes['contour_left2']['x'])
    feature_vector['width_face_cheekbones'] = width_face_cheekbones
    # 3. width of face at mouth
    width_face_mouth = abs(attributes['contour_right5']['x'] - attributes['contour_left5']['x'])
    feature_vector['width_face_mouth'] = width_face_mouth
    # 4. distance from cheek to middle of mouth
    cheek_to_mouth = abs(attributes['contour_right5']['x'] - attributes['mouth_upper_lip_bottom']['x'])
    feature_vector['cheek_to_mouth'] = cheek_to_mouth
    # 5. eye height to length of face ratio
    eye_height = abs(attributes['right_eye_top']['y'] - attributes['right_eye_bottom']['y']) / len_face
    feature_vector['eye_height'] = eye_height
    # 6. eye width to width of face at cheekbones ratio
    eye_width = (abs(attributes['right_eye_right_corner']['x'] - attributes['right_eye_left_corner']['x'])) / width_face_cheekbones
    feature_vector['eye_width'] = eye_width
    # 7. eye area
    eye_area = eye_height * eye_width
    feature_vector['eye_area'] = eye_area
    # 8. width of face at eye level
    width_face_eye = abs(attributes['contour_right1']['x'] - attributes['contour_left1']['x'])
    feature_vector['width_face_eye'] = width_face_eye
    # 9. outer eye width to width of face at eye level ratio
    outer_eye_width = abs(attributes['right_eye_right_corner']['x'] - attributes['left_eye_left_corner']['x']) / width_face_eye
    feature_vector['outer_eye_width'] = outer_eye_width
    # 10. inner eye width to width of face at eye level ratio
    inner_eye_width = abs(attributes['right_eye_left_corner']['x'] - attributes['left_eye_right_corner']['x']) / width_face_eye
    feature_vector['inner_eye_width'] = inner_eye_width
    # 11. nose length
    len_nose = abs((abs(attributes['nose_left']['y'] + attributes['nose_right']['y']) / 2) - attributes['nose_contour_lower_middle']['y']) / len_face
    feature_vector['len_nose'] = len_nose
    # 12. nose tip width
    nose_tip_width = abs(attributes['nose_left']['x'] + attributes['nose_right']['x']) / width_face_mouth
    feature_vector['nose_tip_width'] = nose_tip_width
    # 13. nose area
    nose_area = (len_nose * nose_tip_width) / width_face_mouth
    feature_vector['nose_area'] = nose_area
    # 14. nostril width
    nostril_width = abs(attributes['nose_contour_left3']['x'] - attributes['nose_contour_right3']['x']) / width_face_mouth
    feature_vector['nostril_width'] = nostril_width
    # 15. chin length
    len_chin = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['contour_chin']['y']) / len_face
    feature_vector['len_chin'] = len_chin
    # 16. chin width
    chin_width = abs(attributes['contour_left7']['x'] - attributes['contour_right7']['x']) / len_face
    feature_vector['chin_width'] = chin_width
    # 17. chin area
    chin_area = len_chin * chin_width
    feature_vector['chin_area'] = chin_area
    # 18. eye separation to width of face at cheekbones ratio
    eye_sep = abs(attributes['left_eye_pupil']['x'] - attributes['right_eye_pupil']['x']) / width_face_cheekbones
    feature_vector['eye_sep'] = eye_sep
    # 19. cheekbone prominence
    cheek_prom = abs(width_face_cheekbones - width_face_mouth) / len_face
    feature_vector['cheek_prom'] = cheek_prom
    # 20. cheek thinness (width of cheek to length of face ratio)
    cheek_thin = (abs(attributes['mouth_right_corner']['x'] - attributes['contour_right5']['x'])) / len_face
    feature_vector['cheek_thin'] = cheek_thin
    # 21. facial narrowness (length of face to width of face at mouth ratio)
    facial_narrow = len_face / width_face_mouth
    feature_vector['facial_narrow'] = facial_narrow
    # 22. eyebrow height
    eyebrow_height = abs(attributes['right_eye_pupil']['y'] - attributes['right_eyebrow_lower_middle']['y']) / len_face
    feature_vector['eyebrow_height'] = eyebrow_height
    # 23. upper lip thickness (height of upper lip to length of face ratio)
    up_lip_thick = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['mouth_upper_lip_top']['y']) / len_face
    feature_vector['up_lip_thick'] = up_lip_thick
    # 24. lower lip thickness (height of lower lip to length of face ratio)
    low_lip_thick = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['mouth_lower_lip_bottom']['y']) / len_face
    feature_vector['low_lip_thick'] = low_lip_thick
    # 25. length of lip to width of face at mouth ratio
    len_lip = abs(attributes['mouth_right_corner']['x'] - attributes['mouth_left_corner']['x']) / width_face_mouth
    feature_vector['len_lip'] = len_lip
    # 26. brow_thick (height of brow to length of face ratio)
    brow_thick = abs(attributes['right_eyebrow_lower_middle']['y'] - attributes['right_eyebrow_upper_middle']['y']) / len_face
    feature_vector['brow_thick'] = brow_thick
    # 27a. symmetry of middle of pupil to side of face
    sym1 = abs(attributes['right_eye_pupil']['x'] - attributes['contour_right1']['x']) / abs(attributes['contour_left1']['x'] - attributes['left_eye_pupil']['x'])
    feature_vector['sym1'] = sym1
    # 27b. symmetry of middle of nose to side of face
    sym2 = abs(attributes['nose_tip']['x'] - attributes['contour_right3']['x']) / abs(attributes['contour_left3']['x'] - attributes['nose_tip']['x'])
    feature_vector['sym2'] = sym2
    # 27c. symmetry of center of mouth to side of face
    sym3 = abs(attributes['mouth_upper_lip_bottom']['x'] - attributes['contour_right5']['x']) / abs(attributes['contour_left5']['x'] - attributes['mouth_upper_lip_bottom']['x'])
    feature_vector['sym3'] = sym3
    # 28. distance between centers of pupils
    pupil_width = abs(attributes['left_eye_pupil']['x'] - attributes['right_eye_pupil']['x'])
    feature_vector['pupil_width'] = pupil_width
    # 29. nose width to pupil width ratio
    nose_pupil_width = abs(attributes['nose_left']['x'] - attributes['nose_right']['x']) / pupil_width
    feature_vector['nose_pupil_width'] = nose_pupil_width

    # lipVsPupil = lipLength/pupilWidth
    # noseVsLip = noseTipWidth/lipLength
    # angleCheekboneLowerJaw = 
    # anglePupilLowerJaw1 = 
    # anglePupilLowerJaw2 = 
    # pupilVsNoseWidth = abs(attributes['nose_left']['x'] - attributes['nose_right']['x'])/ ######
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
        eta = 1 / math.sqrt(t + 1)   

        for person_id, attrs in train_data.items():
            feature_vector = extract_features(attrs['attributes'])
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






