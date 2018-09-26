# -*- coding: utf-8 -*-

import os
import pickle
import cv2
import dlib
import numpy as np


def change_contrast_and_brightness(image, alpha=1.0, beta=0):
    image = alpha * image + beta
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype(np.uint8)


predictor_file_path = 'data/models/shape_predictor_68_face_landmarks.dat'
image_folder_path = 'data/myself'
if not os.path.exists(image_folder_path):
    os.mkdir(image_folder_path)
image_id_file_path = os.path.join(image_folder_path, 'image_id.pkl')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_file_path)

if os.path.exists(image_id_file_path):
    with open(image_id_file_path, 'rb') as f:
        image_id = pickle.load(f)
else:
    image_id = 0

image_id_file = open(image_id_file_path, 'wb')
cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        continue
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))
    image = dlib.get_face_chip(img, faces[0], size=64)
    image = change_contrast_and_brightness(image, alpha=np.random.uniform(0.6, 3.0), beta=np.random.randint(-50, 100))
    cv2.imshow('myself', image)
    print(image_id)
    cv2.imwrite(os.path.join(image_folder_path, '%d.jpg' % image_id), image)
    image_id += 1
    if cv2.waitKey(1) == 27 or image_id >= 10000:
        pickle.dump(image_id, image_id_file)
        image_id_file.close()
        break

cv2.destroyAllWindows()
