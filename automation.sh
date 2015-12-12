# File: automation.sh
# --------------------
# Automates the process of creating the dict of faces and running
# our algorithm on the data.

python create_faces.py
python face_data.py train.txt
python face_data.py dev.txt
python face_data.py test.txt
python algorithm.py train_data.txt dev_data.txt

