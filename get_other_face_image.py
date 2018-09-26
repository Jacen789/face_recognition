# -*- coding: utf-8 -*-

import os
import cv2
import dlib

predictor_file_path = 'data/models/shape_predictor_68_face_landmarks.dat'
image_folder_path = 'data/lfw'
image_output_folder_path = 'data/others'
if not os.path.exists(image_output_folder_path):
    os.mkdir(image_output_folder_path)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_file_path)

cnt = 0
for root, dirs, files in os.walk(image_folder_path, topdown=False):
    for filename in files:
        if filename.endswith('.jpg'):
            image_file_path = os.path.join(root, filename)
            img = cv2.imread(image_file_path)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = detector(rgb_image)
            num_faces = len(dets)
            if num_faces == 0:
                print("Sorry, there were no faces found.")
                continue
            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(sp(img, detection))
            images = dlib.get_face_chips(img, faces, size=64)
            for image in images:
                cv2.imshow('others', image)
                cv2.imwrite(os.path.join(image_output_folder_path, filename), image)
                if cv2.waitKey(1) == 27:
                    exit(0)

cv2.destroyAllWindows()
