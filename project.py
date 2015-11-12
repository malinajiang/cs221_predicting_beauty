TEST_IMAGE = 'http://cdn.clubzone.com/content/uploads/artists/2668199/1.jpg'

API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'

from facepp import API

api = API(API_KEY, API_SECRET)

faces = {}

faces_data_file = open('faces-data.txt', 'r')
for line in faces_data_file:
	tokens = line.rstrip('\n').split()
	key = tokens[0]
	faces[key] = {}
	faces[key]['url'] = tokens[1]
	faces[key]['rating'] = tokens[2]
	faces[key]['face_id'] = tokens[3]
	print str(api.detection.landmark(face_id = faces[key]['face_id']))
	break
faces_data_file.close()









# result = api.detection.landmark(face_id = '57a9d4444a50b5ce4c74246ddbb82aa4')
# print result