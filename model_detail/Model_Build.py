import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential

DataSet_Path = '../dataset'

# Define some parameters for the CNN
batch_size = 32  # amount of samples you feed in your network.
img_height = img_width = 224
epochs = 10

train_dataSet = tf.keras.utils.image_dataset_from_directory(
          DataSet_Path,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size)

val_dataSet = tf.keras.utils.image_dataset_from_directory(
          DataSet_Path,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size)


# get and print all classes in our dataset
class_names = train_dataSet.class_names
print("class_names :", class_names)



# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_dataSet.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # keeps the images in memory after they're loaded off disk during the first epoch. then overlaps data preprocessing and model execution while training.
val_ds = val_dataSet.cache().prefetch(buffer_size=AUTOTUNE)  # keeps the images in memory after they're loaded off disk during the first epoch. then overlaps data preprocessing and model execution while training.
normalization_layer = layers.Rescaling(1./255)  # Standardize the data
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# Create the mode
num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])  # Compile the model
model.summary()  # View all the layers of the network using

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Save the model.
model.save('./model/Face_Mask_Detection_Module.h5')
