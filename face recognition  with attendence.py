import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImageBasic/HRITIKA THAKUR.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('ImageBasic/HRITIKA THAKUR TEST.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

cv2.imshow('HRITIKA THAKUR',imgElon)
cv2.imshow('HRITIKA THAKUR TEST',imgtest)
cv2.waitKey(0)