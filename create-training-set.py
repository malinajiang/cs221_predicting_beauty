import pickle


faces_dict_file = open('faces_dict.txt', 'rb')

# dictionary of person_id : {'url', 'rating', 'face_id', 'features'}
faces = pickle.load(faces_dict_file)
faces_dict_file.close()

# training_set_file = open('training_set.txt', 'w')
# for face in faces

print faces