#!/bin/python

import cv2
import numpy
import matplotlib

# Open the webcam Video capture
capture = cv2.VideoCapture(0)
cascPath = "../FaceDetect/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


while (True):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
