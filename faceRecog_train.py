import cv2, os
import numpy as np

# path
rootPath = "../../DATA/"
cascadePath = rootPath+'haarcascades/haarcascade_frontalface_default.xml'
modelPath = rootPath+'trainer'
datasetPath = rootPath+'dataset'

def checkPath(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier(cascadePath);

# Create method to get the images and label data
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    # Initialize empty face sample
    faceSamples=[]

    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        print(imagePath)
        img = cv2.imread(imagePath,0)

        # Get the image id
        file_name = os.path.split(imagePath)[-1]
        id = int(file_name.split(".")[1])

        # Get the face from the training images
        faces = detector.detectMultiScale(img)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels(datasetPath)

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
checkPath(modelPath)
recognizer.save(modelPath+'/trainer.yml')
