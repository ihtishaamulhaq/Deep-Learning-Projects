# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:35:10 2023

@author: Ihtishaam
"""

#	Import the  necessary libraries TensorFlow

import tensorflow as tf
# from tensorflow import keras

# Load the fashion MNIST dataset from directly Keras datasets API. 
Mnist = tf.keras.datasets.fashion_mnist

image_size=28
channel=1
input_image_size=(image_size, image_size, channel)

Desire_accuracy=0.95
print(Desire_accuracy)

class callback_fun(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>Desire_accuracy):
      print("\nReached ", Desire_accuracy*100, "% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = callback_fun()



(Training_images, Training_labels), (Test_images, Test_labels) = Mnist.load_data()

# Training_images reshaping. 
Training_images=Training_images.reshape(60000, 28, 28, 1)

# Training_images normalizing. 
Training_images=Training_images / 255.0

# Test_images reshaping.
Test_images = Test_images.reshape(10000, 28, 28, 1)

# Test_images normalizing.
Test_images=Test_images/255.0

Model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(input_image_size)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])



Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model

Model.summary()

#   Train the model 

Model.fit(Training_images, Training_labels, epochs=10, callbacks=[callbacks])


Test_loss, Test_acc = Model.evaluate(Test_images, Test_labels)

# print the test_acc

print(Test_acc)
