import numpy as np
import imutils
import cv2
import time
from config import config
from track.facetracker import FaceTracker


# Load the open CV model
caffe = cv2.dnn.readNetFromCaffe(config.PROTOTXT_PATH, config.WEIGHTS_PATH)
trackID = []
ft = FaceTracker()

webcam = cv2.VideoCapture(config.VIDEO)
time.sleep(2.0)
if (webcam.isOpened() == False):
    print('\nUnable to read camera feed')


while True:
    success, frame = webcam.read()
    if success == True:
        frame = imutils.resize(frame, 700)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        caffe.setInput(blob)
        detections = caffe.forward()
        rect = []
        
        # Object Detection
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # Top left (beginX, beginY) and bottom right (endX, endY)
                beginX, beginY, endX, endY = box.astype('int')
                rect.append((beginX, beginY, endX, endY))
                cv2.rectangle(frame, (beginX, beginY), (endX, endY), (240, 135, 87), 1)
                
        # Object Tracking
        objectsTracked = ft.update(rect)
        for objectID, centroid in objectsTracked.items():
            # We will append the unique people count to trackID, so 
            # we can display the count in real-time on screen
            trackID.append(objectID)
            # Grab every new ID of faces
            count = (list(set(trackID))[-1])
            text = f"Person {objectID}"
            cv2.putText(frame, text, (centroid[0] - 5, centroid[1] - 5), config.FONT, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            # Get status box
            cv2.rectangle(frame, (700, 5), (493, 40), (0, 255, 0), -1)
            cv2.putText(frame, "Face Tracked: ", (500, 30), config.FONT, 1, (240, 135, 87), 1, cv2.LINE_AA)
            cv2.putText(frame, f" {count}", (659, 31), config.FONT, 1, (240, 135, 87), 1, cv2.LINE_AA)
        # If no face is detected, signify
        if rect == []:
            cv2.rectangle(frame, (700, 5), (489, 40), (0, 255, 0), -1)
            cv2.putText(frame, "No Face Detected", (490, 30), config.FONT, 1, (240, 135, 87), 1, cv2.LINE_AA)
            

        cv2.imshow('Live', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

webcam.release()
cv2.destroyAllWindows()