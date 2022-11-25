import cv2
import numpy as np
from keras.models import load_model


model = load_model('model/Face_Mask_Detection_Module.h5')  # Load the model.

labels = open('model/labels.txt', 'r').readlines()  # Grab the labels from the labels.txt file.


camera = cv2.VideoCapture(0)
camera.set(3, 1080)
camera.set(4, 920)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load the cascade
while True:
    ret, image = camera.read() # Grab the cameras image.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert each video frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect the faces
    # Draw the rectangle around each face


    image_face = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the raw image into (224-height,224-width) pixels.
    image_face = np.asarray(image_face, dtype=np.float32).reshape(1, 224, 224, 3)  # Change the image into numpy array then reshape it to the models input shape.
    image_face = (image_face / 127.5) - 1  # Normalize the image array
    probabilities = model.predict(image_face)  # Have the model predict what the current image is. Model.predict
    print('probabilities Score :', probabilities)
    print('Prediction Outcome  :', labels[np.argmax(probabilities)])  # Print what the highest value probabilitie label

    cv2.imshow('Face Mask Detect', image)  # Show the image in a window

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break




camera.release()
cv2.destroyAllWindows()

