# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:07:08 2024

@author: ghita
"""
from PIL import Image, ImageDraw
import face_recognition 
import cv2

def facial_recognition():
    
    #Load my facial image and get its encoding
    image = face_recognition.load_image_file(r'C:\Users\ghita\OneDrive\Images\miphoto.png')
    face_loc = face_recognition.face_locations(image)
    face_enc = face_recognition.face_encodings(image, face_loc)
    
    #Get frame from camera
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        print("Connected to camera")
        while True :
            capture, frame = cam.read()
            if capture :
                print("frame was captured")
                cam.release()
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                for face_encoding in face_encodings:
                    # Compare the face with known faces
                    matches = face_recognition.compare_faces(face_enc, face_encoding)
                    name = "Unknown"
                    # Find the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(face_enc, face_encoding)
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = "ghita"
                        # Print the name
                    print(name)
                    break
                
            break
    
    