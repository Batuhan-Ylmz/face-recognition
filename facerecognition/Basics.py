# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 23:47:49 2022

@author: Batuhan YILMAZ
"""

import cv2
import numpy as np
import face_recognition

#first step is loading images and converting to rgb, cuz library counts it as bgr
imgElon = face_recognition.load_image_file('ImageBasics/elonmusk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImageBasics/elontest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


#step 2 is finding the faces in images and finding encodings in them

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


#3. step is comparing these faces and finding the distance between them

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)

#final step is to display this on the actual image
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)





cv2.imshow('ElonMusk', imgElon)
cv2.imshow('ElonTest', imgTest)
cv2.waitKey(0)