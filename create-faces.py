# read through text files, create dict of image id to data

# id: {url, rating, face_id}
faces = {}

face_id_file = open('faces-data.txt', 'w')

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


API_KEY = '1291b3a4438bb266a5e8e6a9cc10e15c'
API_SECRET = 'kcPlNG7tYo6ZRiAhCC_a1tNHWJyx1odN'


from facepp import API

api = API(API_KEY, API_SECRET)

print 'Running face detection...'
for face in faces:
	print face
	url = faces[face]['url']
	faces[face]['face_id'] = api.detection.detect(url = url)['face'][0]['face_id']
	face_id_file.write(' '.join([face, faces[face]['url'], faces[face]['rating'], faces[face]['face_id']]) + '\n')



face_id_file.close()

print 'finished!'
