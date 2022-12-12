import cv2
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np
import tensorflow as tf
import dlib

model = load_model('model_detail/model/Face_Mask_Detection_Module.h5')  # Load the model
labels = open('model_detail/model/labels.txt', 'r').readlines()  # Grab the labels from the labels.txt file.
face_detector = dlib.get_frontal_face_detector() # initialize the HOG face detector

back_ground_color = 'white'  # UI background color

def detect_mask(faces, frame):
    # loop over the faces and draw a rectangle around each face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        try:
            face_crop = frame[y:y + h, x:x + w]
            image_face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the raw image into (224-height,224-width) pixels.
            image_face = np.asarray(image_face, dtype=np.float32).reshape(1, 224, 224, 3)  # Change the image into numpy array then reshape it to the models input shape.
            image_face = (image_face / 127.5) - 1  # Normalize the image array
        except:
            show_frame()
        predictions = model.predict(image_face)
        score = tf.nn.softmax(predictions[0])
        Class = labels[np.argmax(score)]
        Confidence_score = int(100 * np.max(score))
        Probability_score.config(text=f"{Confidence_score} %")
        if Class[0] == '0':
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f'With MASK :{Confidence_score}%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            Face_Mask_status.config(text='With MASK', fg='green')
        if Class[0] == '1':
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, f'Without MASK ::{Confidence_score}%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            Face_Mask_status.config(text='Without MASK', fg='red')


root = tk.Tk()  # UI
root.title("Face Mask Detection (group 3 : computer vision)")
root.minsize(1000, 950)

content_frame = tk.Frame(root, bg=back_ground_color)
content_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

tk.Label(content_frame, text='FACE MASK STATUS :', bg=back_ground_color, anchor='e', font=('Consolas', 11, 'bold')).place(relheight=0.04, relwidth=0.23, rely=0.75, relx=0.02)
Face_Mask_status = tk.Label(content_frame, bg=back_ground_color, anchor='w', font=('Courier New', 12, 'bold'))
Face_Mask_status.place(relheight=0.04, relwidth=0.25, rely=0.75, relx=0.251)

tk.Label(content_frame, text='PROBABILITY SCORE ', bg=back_ground_color, anchor='e', font=('Consolas', 11, 'bold')).place(relheight=0.04, relwidth=0.23, rely=0.785, relx=0.02)
Probability_score = tk.Label(content_frame, bg=back_ground_color, fg='blue', anchor='w', font=('Courier New', 12, 'bold'))
Probability_score.place(relheight=0.04,  relwidth=0.25, rely=0.785, relx=0.251)


v_frame = tk.Frame(content_frame, bg=back_ground_color)
v_frame.place(relx=0.05, rely=0.02, relwidth=0.9, relheight=0.7)

v_Lable = tk.Label(v_frame, bg=back_ground_color)
v_Lable.place(relheight=1, relwidth=1, rely=0, relx=0)

cap_video = cv2.VideoCapture(0)  # Capture video frame
cap_video.set(3, 1080)
cap_video.set(4, 920)


def show_frame():
    _, frame = cap_video.read()  # Read the frame
    faces = face_detector(frame, 0) # detect faces in the frame
    detect_mask(faces, frame)

    # Display video frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    v_Lable.imgtk = imgtk
    v_Lable.configure(image=imgtk)
    v_Lable.after(5, show_frame)


show_frame()
root.mainloop()
