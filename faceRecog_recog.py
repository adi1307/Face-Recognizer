import cv2
import numpy as np
import os

def checkPath(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# path
rootPath = "../../DATA/"
cascadePath = rootPath+'haarcascades/haarcascade_frontalface_default.xml'
modelPath = rootPath+'trainer'
datasetPath = rootPath+'dataset'
checkPath(modelPath)

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read(modelPath+'/trainer.yml')

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(1)

# Loop
while True:
    # Read the video frame
    ret, frame =cam.read()
    frame = cv2.flip(frame,1)

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist
        if(Id == 1):
            Id = "ayushi {0:.2f}%".format(round(100 - confidence, 2))
        elif(Id == 2):
            Id = "rajat {0:.2f}%".format(round(100 - confidence, 2))

        # Put text describe who is in the picture
        cv2.putText(frame, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('frame',frame)

    # If 'q' is pressed, close program
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
