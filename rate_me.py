# File: rate_me.py
# --------------------
# Given a photo of a face, returns an attractiveness rating.

import math
from algorithm import *
from util import *
import sys
from facepp import API

API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'

api = API(API_KEY, API_SECRET)

def rate_face(url, weights):
	face_id = api.detection.detect(url = url)['face'][0]['face_id']
	attributes = api.detection.landmark(face_id = face_id)['result'][0]['landmark']
	
	feature_vector = extract_features(attributes)
	rating = dot_product(weights, feature_vector)
	
	return rating

def main(argv):
	weights = learn_predictor("train_data.txt", None, extract_features, False)
	rating = rate_face(argv[0], weights)
	print "rating : %s" % rating

if __name__ == "__main__":
    main(sys.argv[1:])
    # main("http://imageshack.com/a/img911/699/FpqkvC.jpg")