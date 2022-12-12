import cv2
import dlib
import tensorflow as tf
from keras.models import load_model
import numpy as np

model = load_model('model_detail/model/Face_Mask_Detection_Module.h5')
labels = open('model_detail/model/labels.txt', 'r').readlines()

vs = cv2.VideoCapture(0)# initialize the video stream and allow the camera sensor to warmup

# initialize the HOG face detector
face_detector = dlib.get_frontal_face_detector()

def detect_mask(faces, frame):
    # loop over the faces and draw a rectangle around each face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        face_crop = frame[y:y + h, x:x + w]
        image_face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the raw image into (224-height,224-width) pixels.
        image_face = np.asarray(image_face, dtype=np.float32).reshape(1, 224, 224, 3)  # Change the image into numpy array then reshape it to the models input shape.
        image_face = (image_face / 127.5) - 1  # Normalize the image array

        predictions = model.predict(image_face)
        score = tf.nn.softmax(predictions[0])
        Class = labels[np.argmax(score)]
        Confidence_score = int(100 * np.max(score))
        if Class[0] == '0':
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f'With MASK :{Confidence_score}%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if Class[0] == '1':
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, f'Without MASK ::{Confidence_score}%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


while True:
    _, frame = vs.read() # grab the current frame
    faces = face_detector(frame, 0) # detect faces in the frame
    detect_mask(faces, frame)

    # show the frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
