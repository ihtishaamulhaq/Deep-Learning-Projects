# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:35:10 2023

@author: Ihtishaam
"""

#	Import the  necessary libraries TensorFlow

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


# Load the fashion MNIST dataset from directly Keras datasets API. 
Mnist = tf.keras.datasets.fashion_mnist

image_size=28
channel=1
input_image_size=(image_size, image_size, channel)

Desire_accuracy=0.96
print(Desire_accuracy)

class callback_fun(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>Desire_accuracy):
      print("\nReached ", Desire_accuracy*100, "% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = callback_fun()

(Training_images, Training_labels), (Test_images, Test_labels) = Mnist.load_data()

# First, split the data into training and temporary (temp) data
Train_images, Temp_images, Train_labels, Temp_labels = train_test_split(Training_images, Training_labels, test_size=0.3, random_state=42)

# Then, split the temp data into validation and test data
Val_images, Test_images1, Val_labels, Test_labels1 = train_test_split(Temp_images, Temp_labels, test_size=0.5, random_state=42)

# Training_images reshaping. 
# Train_images=Training_images.reshape(60000, 28, 28, 1)

# Training_images normalizing. 
Train_images=Train_images / 255.0

# Test_images reshaping.
# Test_images = Test_images.reshape(10000, 28, 28, 1)

# Test_images normalizing.
Test_images1=Test_images1/255.0

# Test_images reshaping.
# Val_images = Val_images.reshape(10000, 28, 28, 1)

# Test_images normalizing.
Val_images=Val_images/255.0

Model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
                         input_shape=(input_image_size)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Model Summary
Model.summary()

"""
compile the model with well known optimizer
Sparse Categorical Crossentropy loss is used because 
of multi-class classification
"""
Model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#  Train the model 
EPOCHS=10
history=Model.fit(Train_images, 
                  Train_labels, 
                  epochs=EPOCHS,
                  validation_data=(Val_images, Val_labels),
                  callbacks=[callbacks])

# Get the accuracy and loss during tarining
Accuracy=history.history['accuracy']
loss=history.history['loss']
validation_loss=history.history['val_loss']
validation_Accuracy=history.history['val_accuracy']

"""
plot the accuracy and validation accuracy graph
"""
plt.figure(figsize=(8,8))
plt.plot(range(EPOCHS), Accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), validation_Accuracy, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')
plt.savefig("training_and_validation_accuracy.png")
plt.show()

"""
plot the validation loss and training loss graph
"""

plt.figure(figsize=(8,8))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("training_and_validation_loss.png")
plt.show()

"""
preddiction on test data
"""
prediction=Model.predict(Test_images1)
y_predict=np.argmax(prediction, axis=1)

# compute the confusiom matrix
cm=confusion_matrix(Test_labels1, y_predict)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 
               'Dress', 'Coat', 'Sandal', 'Shirt', 
               'Sneaker', 'Bag', 'Ankle boot']
# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", 
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues", cbar=False)
plt.xlabel("Actual ")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.savefig("Confusion_matrix.png")
plt.show()