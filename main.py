import cv2
import numpy as np
import face_recognition

imgbill = face_recognition.load_image_file('ImageBasic/HRITIKA THAKUR.jpg')
imgbill = cv2.cvtColor(imgbill,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('ImageBasic/HRITIKA THAKUR.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgbill)[0]
encodebill = face_recognition.face_encodings(imgbill)[0]
cv2.rectangle(imgbill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodebill],encodetest)
faceDis = face_recognition.face_distance([encodebill],encodetest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('HRITIKA THAKUR',imgbill)
cv2.imshow('HRITIKA THAKUR test',imgtest)
cv2.waitKey(0)