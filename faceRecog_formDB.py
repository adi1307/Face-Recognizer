import cv2
import os

# path for storing the datasets
rootPath = "../../DATA/"
cascadePath = rootPath+'haarcascades/haarcascade_frontalface_default.xml'
datasetPath = rootPath+'dataset'

def checkPath(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start capturing video
vid_cam = cv2.VideoCapture(1)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(cascadePath)

# For each person, one face id
face_name = input("Enter Name: ")
face_id = int(input("Enter face ID (integer): "))

# Initialize sample face image
count = 0

checkPath(datasetPath)

# Start looping
while(True):

    # Capture video frame
    _, frame = vid_cam.read()
    frame = cv2.flip(frame,1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        name = datasetPath + '/' + face_name + '.' + str(face_id) + '.' + str(count) + ".jpg"
        cv2.imwrite(name, gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>100:
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
