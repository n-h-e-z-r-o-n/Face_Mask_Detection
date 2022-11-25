import cv2
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

model = load_model('model_detail/model/Face_Mask_Detection_Module.h5')  # Load the model
labels = open('model_detail/model/labels.txt', 'r').readlines()  # Grab the labels from the labels.txt file.
back_ground_color = 'white'  # UI background color

detector = MTCNN()  # face detector
root = tk.Tk()  # UI
root.title("Face Mask Detection (group 3 : computer vision)")
root.minsize(1000, 950)

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
        if Confidence_score >= 60:
            color = "Green"
        else:
            color = "red"
        Probability_score.config(text=f"{Confidence_score} %", fg=color)
        if Class[0] == '0':
            cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 3)
            cv2.putText(image, f'With MASK :{Confidence_score}%', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            Face_Mask_status.config(text="Mask On", fg='Green')
        if Class[0] == '1':
            cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 3)
            cv2.putText(image, f'Without MASK ::{Confidence_score}%', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            Face_Mask_status.config(text="No Mask On", fg='Brown')

def extract_faces(result_list, img):
    face_s = []
    box_s = []
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = width + x1
        y2 = height + y1
        crop = img[y1:y1 + y2, x1:x1 + x2]  # img[y1:y2, x1:x2, :]
        image_face = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the raw image into (224-height,224-width) pixels.
        image_face = np.asarray(image_face, dtype=np.float32).reshape(1, 224, 224, 3)  # Change the image into numpy array then reshape it to the models input shape.
        image_face = (image_face / 127.5) - 1  # Normalize the image array
        face_s.append(image_face)
        box_s.append(result_list[i])
    return box_s, face_s



content_frame = tk.Frame(root, bg=back_ground_color)
content_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

tk.Label(content_frame, text='FACE MASK STATUS :', bg=back_ground_color, anchor='e', font=('Consolas', 11, 'bold')).place(relheight=0.04, relwidth=0.23, rely=0.75, relx=0.02)
Face_Mask_status = tk.Label(content_frame, bg=back_ground_color, anchor='w', font=('Courier New', 12, 'bold'))
Face_Mask_status.place(relheight=0.04, relwidth=0.25, rely=0.75, relx=0.251)

tk.Label(content_frame, text='PROBABILITY SCORE ', bg=back_ground_color, anchor='e', font=('Consolas', 11, 'bold')).place(relheight=0.04, relwidth=0.23, rely=0.785, relx=0.02)
Probability_score = tk.Label(content_frame, bg=back_ground_color, fg='white', anchor='w', text='with',font=('Courier New', 12, 'bold'))
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
    face_boxes = detector.detect_faces(frame)
    if face_boxes:
        boxes, faces = extract_faces(face_boxes, frame)
        draw_on_face(frame, boxes, faces)

    # Display video frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    v_Lable.imgtk = imgtk
    v_Lable.configure(image=imgtk)
    v_Lable.after(5, show_frame)

show_frame()
root.mainloop()
