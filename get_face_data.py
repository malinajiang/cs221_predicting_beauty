# File: get_face_data.py 
# --------------------
# Gets attributes data for each face in the training set and pickles into
# dict of image_id to data.

import pickle

API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'

from facepp import API

api = API(API_KEY, API_SECRET)

faces = {}
total_faces = 597
index = 1

# print total_faces
training_data_file = open('training_set.txt', 'r')
for line in training_data_file:
	if index > 50: break

	print 'Working on face ' + str(index) + '/' + str(total_faces) + ': ',

	tokens = line.rstrip('\n').split()
	key = tokens[0]
	print key
	
	faces[key] = {}
	faces[key]['url'] = tokens[1]
	faces[key]['rating'] = tokens[2]
	faces[key]['face_id'] = tokens[3]
	faces[key]['attributes'] = api.detection.landmark(face_id = faces[key]['face_id'])

	index += 1

training_data_file.close()

face_dict_file = open('face_dict.txt', 'wb')
pickle.dump(faces, face_dict_file)
face_dict_file.close()
