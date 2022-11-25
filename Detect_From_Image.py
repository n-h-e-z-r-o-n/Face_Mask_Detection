from mtcnn import MTCNN
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model


model = load_model('model_detail/model/Face_Mask_Detection_Module.h5')
labels = open('model_detail/model/labels.txt', 'r').readlines()
detector = MTCNN()


def draw_on_face(image, f_box, faces):
    for i in range(len(f_box)):
        x1 = f_box[i]['box'][0]
        y1 = f_box[i]['box'][1]
        x2 = f_box[i]['box'][2]
        y2 = f_box[i]['box'][3]

        predictions = model.predict(faces[i])
        score = tf.nn.softmax(predictions[0])
        Class = labels[np.argmax(score)]
        Confidence_score = int(100 * np.max(score))

        if Class[0] == '0':
            cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 3)
            cv2.putText(image, f'With MASK :{Confidence_score}%', (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        if Class[0] == '1':
            cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 3)
            cv2.putText(image, f'Without MASK ::{Confidence_score}%', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imwrite('filename.png', image)


def extract_faces(result_list, img):
    face_s = []
    box_s = []
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = width + x1
        y2 = height + y1

        crop = img[y1:y1 + y2, x1:x1 + x2] # img[y1:y2, x1:x2, :]
        image_face = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the raw image into (224-height,224-width) pixels.
        cv2.imwrite('g.png', image_face)
        image_face = np.asarray(image_face, dtype=np.float32).reshape(1, 224, 224, 3)  # Change the image into numpy array then reshape it to the models input shape.
        image_face = (image_face / 127.5) - 1  # Normalize the image array
        face_s.append(image_face)
        box_s.append(result_list[i])
        print(i)

    return box_s, face_s


Image_Path = "C:/Users/HEZRON WEKESA/Pictures/IMG_20210325_000150.jpg"
test_img_path = Image_Path
img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
face_boxes = detector.detect_faces(img)
boxes, faces = extract_faces(face_boxes, img)
draw_on_face(img, boxes, faces)

