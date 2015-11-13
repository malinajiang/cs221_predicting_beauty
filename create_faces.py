# File: create_faces.py
# --------------------
# Reads through text files and creates dict of image id to data.

import sys
from facepp import API

API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'

api = API(API_KEY, API_SECRET)

def create_faces():
	# image_id: {url, rating, face_id}
	faces = {}

	face_ids_file = open('face_ids.txt', 'w')

	# image_urls = {}
	image_urls_file = open('image_urls.txt', 'r')
	for line in image_urls_file:
		tokens = line.rstrip('\n').split()
		faces[tokens[0]] = {'url': tokens[1]}
	image_urls_file.close()

	# ratings = {}
	ratings_file = open('ratings.txt', 'r')
	for line in ratings_file:
		tokens = line.rstrip('\n').split()
		faces[tokens[0]]['rating'] = tokens[1]
	ratings_file.close()

	print 'Running face detection...'
	for face in faces:
		print face
		url = faces[face]['url']
		faces[face]['face_id'] = api.detection.detect(url = url)['face'][0]['face_id']
		face_ids_file.write(' '.join([face, faces[face]['url'], faces[face]['rating'], faces[face]['face_id']]) + '\n')

	face_ids_file.close()

	print 'finished!'

def main(argv):
	create_faces(argv[0], argv[1])

if __name__ == '__main__':
	main(sys.argv[1:])
