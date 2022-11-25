from keras.models import load_model  # install Keras
from PIL import Image, ImageOps  # Install pillow for image preprocessing
import numpy as np  # Install nampy for math


np.set_printoptions(suppress=True)  # Disable scientific notation for clarity

model = load_model('model/Face_Mask_Detection_Module.h5', compile=False)  # Load the model
class_names = open('./model/labels.txt', 'r').readlines()  # Load the labels

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open('model/test_Image_2.jpg').convert('RGB')  # Test image path
size = (224, 224)  # specifing image resize size to  224x224
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)  # resizing the image to be at least 224x224 and then cropping from the center
image_array = np.asarray(image)  # turn the image into a numpy array
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1  # Normalize the image
data[0] = normalized_image_array  # Load the image into the array

# run the inference
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print the outcome
print('Class:', class_name, end='')
print('Confidence score:', confidence_score)
