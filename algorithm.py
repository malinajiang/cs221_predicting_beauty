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
    # # 3. width of face at mouth
    # width_face_mouth = abs(attributes['contour_right5']['x'] - attributes['contour_left5']['x'])
    # feature_vector['width_face_mouth'] = width_face_mouth
    # 4. distance from cheek to middle of mouth
    cheek_to_mouth = abs(attributes['contour_right5']['x'] - attributes['mouth_upper_lip_bottom']['x'])
    feature_vector['cheek_to_mouth'] = cheek_to_mouth

    eye_h = abs(attributes['right_eye_top']['y'] - attributes['right_eye_bottom']['y'])
    feature_vector['eye_h'] = eye_h

    # # 5. eye height to length of face ratio
    # eye_height = abs(attributes['right_eye_top']['y'] - attributes['right_eye_bottom']['y']) / len_face
    # feature_vector['eye_height'] = eye_height

    eye_w = abs(attributes['right_eye_right_corner']['x'] - attributes['right_eye_left_corner']['x'])
    feature_vector['eye_w'] = eye_w

    # # 6. eye width to width of face at cheekbones ratio
    # eye_width = (abs(attributes['right_eye_right_corner']['x'] - attributes['right_eye_left_corner']['x'])) / width_face_cheekbones
    # feature_vector['eye_width'] = eye_width
    # # 7. eye area
    # eye_area = eye_height * eye_width
    # feature_vector['eye_area'] = eye_area
    # 8. width of face at eye level
    width_face_eye = abs(attributes['contour_right1']['x'] - attributes['contour_left1']['x'])
    feature_vector['width_face_eye'] = width_face_eye

    outer_eye_w = attributes['right_eye_right_corner']['x'] - attributes['left_eye_left_corner']['x']
    feature_vector['outer_eye_w'] = outer_eye_w

    # # 9. outer eye width to width of face at eye level ratio
    # outer_eye_width = abs(attributes['right_eye_right_corner']['x'] - attributes['left_eye_left_corner']['x']) / width_face_eye
    # feature_vector['outer_eye_width'] = outer_eye_width

    inner_eye_w = attributes['right_eye_left_corner']['x'] - attributes['left_eye_right_corner']['x']
    feature_vector['inner_eye_w'] = inner_eye_w

    # 10. inner eye width to width of face at eye level ratio
    inner_eye_width = abs(attributes['right_eye_left_corner']['x'] - attributes['left_eye_right_corner']['x']) / width_face_eye
    feature_vector['inner_eye_width'] = inner_eye_width

    nose_l = abs((abs(attributes['nose_left']['y'] + attributes['nose_right']['y']) / 2) - attributes['nose_contour_lower_middle']['y'])
    feature_vector['nose_l'] = nose_l

    # 11. nose length
    len_nose = abs((abs(attributes['nose_left']['y'] + attributes['nose_right']['y']) / 2) - attributes['nose_contour_lower_middle']['y']) / len_face
    feature_vector['len_nose'] = len_nose

    nose_t = abs(attributes['nose_left']['x'] + attributes['nose_right']['x'])
    feature_vector['nose_t'] = nose_t

    # # 12. nose tip width
    # nose_tip_width = abs(attributes['nose_left']['x'] + attributes['nose_right']['x']) / width_face_mouth
    # feature_vector['nose_tip_width'] = nose_tip_width
    # # 13. nose area
    # nose_area = (len_nose * nose_tip_width) / width_face_mouth
    # feature_vector['nose_area'] = nose_area

    nose_w = abs(attributes['nose_contour_left3']['x'] - attributes['nose_contour_right3']['x'])
    feature_vector['nose_w'] = nose_w

    # # 14. nostril width
    # nostril_width = abs(attributes['nose_contour_left3']['x'] - attributes['nose_contour_right3']['x']) / width_face_mouth
    # feature_vector['nostril_width'] = nostril_width

    l_chin = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['contour_chin']['y'])
    feature_vector['l_chin'] = l_chin

    # # 15. chin length
    # len_chin = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['contour_chin']['y']) / len_face
    # feature_vector['len_chin'] = len_chin

    c_width = abs(attributes['contour_left7']['x'] - attributes['contour_right7']['x'])
    feature_vector['c_width'] = c_width

    # # 16. chin width
    # chin_width = abs(attributes['contour_left7']['x'] - attributes['contour_right7']['x']) / len_face
    # feature_vector['chin_width'] = chin_width    
    # # 17. chin area
    # chin_area = len_chin * chin_width
    # feature_vector['chin_area'] = chin_area

    eye_s = abs(attributes['left_eye_pupil']['x'] - attributes['right_eye_pupil']['x'])
    feature_vector['eye_s'] = eye_s

    # # 18. eye separation to width of face at cheekbones ratio
    # eye_sep = abs(attributes['left_eye_pupil']['x'] - attributes['right_eye_pupil']['x']) / width_face_cheekbones
    # feature_vector['eye_sep'] = eye_sep

    # cheekbone_prom = abs(width_face_cheekbones - width_face_mouth)
    # feature_vector['cheekbone_prom'] = cheekbone_prom

    # # 19. cheekbone prominence
    # cheek_prom = abs(width_face_cheekbones - width_face_mouth) / len_face
    # feature_vector['cheek_prom'] = cheek_prom

    # cheek_t = (abs(attributes['mouth_right_corner']['x'] - attributes['contour_right5']['x']))
    # feature_vector['cheek_t'] = cheek_t

    # # 20. cheek thinness (width of cheek to length of face ratio)
    # cheek_thin = (abs(attributes['mouth_right_corner']['x'] - attributes['contour_right5']['x'])) / len_face
    # feature_vector['cheek_thin'] = cheek_thin
    # 21. facial narrowness (length of face to width of face at mouth ratio)
    # facial_narrow = len_face / width_face_mouth
    # feature_vector['facial_narrow'] = facial_narrow

    eyebrow_h = abs(attributes['right_eye_pupil']['y'] - attributes['right_eyebrow_lower_middle']['y'])
    feature_vector['eyebrow_h'] = eyebrow_h

    # 22. eyebrow height
    eyebrow_height = abs(attributes['right_eye_pupil']['y'] - attributes['right_eyebrow_lower_middle']['y']) / len_face
    feature_vector['eyebrow_height'] = eyebrow_height

    up_lip_t = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['mouth_upper_lip_top']['y'])
    feature_vector['up_lip_t'] = up_lip_t

    # # 23. upper lip thickness (height of upper lip to length of face ratio)
    # up_lip_thick = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['mouth_upper_lip_top']['y']) / len_face
    # feature_vector['up_lip_thick'] = up_lip_thick

    low_lip_t = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['mouth_lower_lip_bottom']['y'])
    feature_vector['low_lip_t'] = low_lip_t

    # 24. lower lip thickness (height of lower lip to length of face ratio)
    low_lip_thick = abs(attributes['mouth_upper_lip_bottom']['y'] - attributes['mouth_lower_lip_bottom']['y']) / len_face
    feature_vector['low_lip_thick'] = low_lip_thick

    len_l = abs(attributes['mouth_right_corner']['x'] - attributes['mouth_left_corner']['x'])
    feature_vector['len_l'] = len_l

    # # 25. length of lip to width of face at mouth ratio
    # len_lip = abs(attributes['mouth_right_corner']['x'] - attributes['mouth_left_corner']['x']) / width_face_mouth
    # feature_vector['len_lip'] = len_lip

    brow_t = abs(attributes['right_eyebrow_lower_middle']['y'] - attributes['right_eyebrow_upper_middle']['y'])
    feature_vector['brow_t'] = brow_t


    # # 27a. symmetry of middle of pupil to side of face

    # s1a = abs(attributes['right_eye_pupil']['x'] - attributes['contour_right1']['x'])
    # feature_vector['s1a'] = s1a

    # s1b = abs(attributes['contour_left1']['x'] - attributes['left_eye_pupil']['x'])
    # feature_vector['s1b'] = s1b

    # sym1 = abs(attributes['right_eye_pupil']['x'] - attributes['contour_right1']['x']) / abs(attributes['contour_left1']['x'] - attributes['left_eye_pupil']['x'])
    # feature_vector['sym1'] = sym1

    # s2a = abs(attributes['nose_tip']['x'] - attributes['contour_right3']['x'])
    # feature_vector['s2a'] = s2a

    # s2b = abs(attributes['contour_left3']['x'] - attributes['nose_tip']['x'])
    # feature_vector['s2b'] = s2b

    # # 27b. symmetry of middle of nose to side of face
    # sym2 = abs(attributes['nose_tip']['x'] - attributes['contour_right3']['x']) / abs(attributes['contour_left3']['x'] - attributes['nose_tip']['x'])
    # feature_vector['sym2'] = sym2

    # s3a = abs(attributes['mouth_upper_lip_bottom']['x'] - attributes['contour_right5']['x'])
    # feature_vector['s3a'] = s3a

    # s3b = abs(attributes['contour_left5']['x'] - attributes['mouth_upper_lip_bottom']['x'])
    # feature_vector['s3b'] = s3b

    # # 27c. symmetry of center of mouth to side of face
    # sym3 = abs(attributes['mouth_upper_lip_bottom']['x'] - attributes['contour_right5']['x']) / abs(attributes['contour_left5']['x'] - attributes['mouth_upper_lip_bottom']['x'])
    # feature_vector['sym3'] = sym3
    # 28. distance between centers of pupils
    # pupil_width = abs(attributes['left_eye_pupil']['x'] - attributes['right_eye_pupil']['x'])
    # feature_vector['pupil_width'] = pupil_width

    nose_w = abs(attributes['nose_left']['x'] - attributes['nose_right']['x'])
    feature_vector['nose_w'] = nose_w

    # # 29. nose width to pupil width ratio
    # nose_pupil_width = abs(attributes['nose_left']['x'] - attributes['nose_right']['x']) / pupil_width
    # feature_vector['nose_pupil_width'] = nose_pupil_width

    # # 30. sym1 squared
    # feature_vector['sym1squared'] = sym1**2

    # # 31. sym2 squared
    # feature_vector['sym2squared'] = sym2**2

    # # 32. sym3 squared
    # feature_vector['sym3squared'] = sym3**2

    # # 33. sym1/sym2
    # feature_vector['sym1sym2'] = sym1*sym2

    # # 34. sym2/sym3
    # feature_vector['sym2sym3'] = sym2*sym3

    # # 35. sym1/sym3
    # feature_vector['sym1sym3'] = sym1*sym3

    # # 36. eye height squared
    # feature_vector['eye_heightsquared'] = eye_height**2

    # # 37. eye width squared
    # feature_vector['eye_widthsquared'] = eye_width**2

    # # 38. eye area squared
    # feature_vector['eye_areasquared'] = eye_area**2

    # # 39. eye height/eye width
    # feature_vector['eye_height/eye_width'] = eye_height*eye_width

    # # 40. eye width/eye area
    # feature_vector['eye_width/eye_area'] = eye_width*eye_area

    # # 41. eye height/eye area
    # feature_vector['eye_height/eye_area'] = eye_height*eye_area

    #pupil_width, len_face, len_lip, eye_sep, width_face_eye, eyebrow_height, width_face_mouth, facial_narrow
    #width_face_eye->len_lip, eye_sep, width_face_mouth, facial_narrow

    # # 42. width_face_eye squared
    # feature_vector['width_face_eyesquared'] = width_face_eye**2

    # # 43. len_lip squared
    # feature_vector['len_lipsquared'] = len_lip**2

    # # 44. eye_sep squared
    # feature_vector['eye_sepsquared'] = eye_sep**2

    # # 45. width_face_mouth squared
    # feature_vector['width_face_mouthsquared'] = width_face_mouth**2

    # # 46. facial_narrow squared
    # feature_vector['facial_narrowsquared'] = facial_narrow**2

    # # 47. width_face_eye/len_lip
    # feature_vector['width_face_eye/len_lip'] = width_face_eye*len_lip

    # # 48. width_face_eye/eye_sep
    # feature_vector['width_face_eye/eye_sep'] = width_face_eye*eye_sep

    # # 49. width_face_eye/width_face_mouth
    # feature_vector['width_face_eye/width_face_mouth'] = width_face_eye*width_face_mouth

    # # 50. width_face_eye/facial_narrow
    # feature_vector['width_face_eye/facial_narrow'] = width_face_eye*facial_narrow
    return feature_vector


def learn_predictor(train, dev, feature_extractor):
    train_data = read_file(train)
    dev_data = read_file(dev)

    weights = collections.defaultdict(lambda: 0)  # feature => weight
    num_iters = 500

    for t in range(num_iters):
        eta = .00001 #0.00001 - 0.000001 
        for person_id, attrs in train_data.items():
            feature_vector = extract_features(attrs['attributes'])
            rating = float(attrs['rating'])
            margin = dot_product(weights, feature_vector) - rating
            increment(weights, -eta * 2 * margin, feature_vector)

        # train_correct = evaluate_predictor(train_data, extract_features, weights)
        # dev_correct = evaluate_predictor(dev_data, extract_features, weights)
        # print "Official: train = %s, dev = %s" % (train_correct, dev_correct)

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
    learn_predictor(argv[0], argv[1], extract_features)

if __name__ == "__main__":
    main(sys.argv[1:])

