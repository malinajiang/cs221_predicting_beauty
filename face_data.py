# File: face_data.py 
# --------------------
# Gets attributes data for each face in the data sets and pickles into
# dict of image_id to data.

import pickle
import sys
from facepp import API

API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'

api = API(API_KEY, API_SECRET)

def retrieve_data(file_name):
	faces = {}
	index = 1

	input_file = open(file_name, 'r')
	for line in input_file:
		# temporary: face++ crashes frequently
		# if index > 50: break

		print 'Working on face ' + str(index) + ': ',

		tokens = line.rstrip('\n').split()
		key = tokens[0]
		print key
		
		faces[key] = {}
		faces[key]['url'] = tokens[1]
		faces[key]['rating'] = tokens[2]
		faces[key]['face_id'] = tokens[3]
		faces[key]['attributes'] = api.detection.landmark(face_id = faces[key]['face_id'])['result'][0]['landmark']

		index += 1

	input_file.close()

	data_file = open(file_name.replace('.txt', '') + '_data.txt', 'wb')
	pickle.dump(faces, data_file)
	data_file.close()

def main(argv):
	retrieve_data(argv[0])

if __name__ == '__main__':
	main(sys.argv[1:])

	