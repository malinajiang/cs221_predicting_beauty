import pickle


TEST_IMAGE = 'http://cdn.clubzone.com/content/uploads/artists/2668199/1.jpg'

API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'

from facepp import API

api = API(API_KEY, API_SECRET)

faces = {}

index = 1
total_faces = 597

# print total_faces
faces_data_file = open('faces-data.txt', 'r')
for line in faces_data_file:
	if (index > 50): break
	print 'Working on face ' + str(index) + '/' + str(total_faces) + ': ',

	tokens = line.rstrip('\n').split()
	key = tokens[0]
	print key
	faces[key] = {}
	faces[key]['url'] = tokens[1]
	faces[key]['rating'] = tokens[2]
	faces[key]['face_id'] = tokens[3]
	faces[key]['features'] = api.detection.landmark(face_id = faces[key]['face_id'])
	
	index += 1
faces_data_file.close()

faces_dict_file = open('faces_dict.txt', 'wb')
pickle.dump(faces, faces_dict_file)
faces_dict_file.close()

# print faces

# unpickle
# faces_dict_file = open('faces_dict.txt', 'rb')
# faces = pickle.load(faces_dict_file)
# faces_dict_file.close()








# result = api.detection.landmark(face_id = '57a9d4444a50b5ce4c74246ddbb82aa4')
# print result