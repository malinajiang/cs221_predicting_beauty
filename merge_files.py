# File: merge_files.py
# --------------------
# Combines all the face data from the train, dev, and test files
# into one dictionary of faces and saves it to all_data.txt.

from util import *

data_file = open('all_data.txt', 'wb')
d1 = read_file('train_data.txt')
d2 = read_file('dev_data.txt')
d3 = read_file('test_data.txt')

for key in d2: d1[key] = d2[key]
for key in d3: d1[key] = d3[key]

pickle.dump(d1, data_file)
data_file.close()

