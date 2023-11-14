# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:54:15 2023

@author: Ihtishaam
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

heart_data=pd.read_csv('E:\PythonCode\Heart Attack analysis & prediction\heart.csv')
print(heart_data.head())

# number of positive and negative cases
sns.countplot(x='output', data=heart_data)
plt.savefig('distribution_plot.png')
plt.show()

# split the indepdent and dependent variable
independent_variable=heart_data.drop('output', axis=1)
print(independent_variable.head())
target=heart_data['output']
print(target)


X_train, X_test, y_train, y_test = train_test_split(independent_variable,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=2)

# Standardize the features (optional but often helpful)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Keras model for binary classification
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')  
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history=model.fit(X_train, y_train, 
                  epochs=30, batch_size=32,
                  validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
accuracy = accuracy_score(y_test, y_pred_binary)
classification_rep = classification_report(y_test, y_pred_binary)
confusion = confusion_matrix(y_test, y_pred_binary)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("Confusion_matrix.png")
plt.show()

# Extract accuracy and validation accuracy from the training history
accuracy=history.history['accuracy']
loss=history.history['loss']
validation_loss=history.history['val_loss']
Validation_accuracy=history.history['val_accuracy']

"""
plot the accuracy and validation accuracy graph
"""
plt.figure(figsize=(8,8))
plt.plot(accuracy, label='Training Accuracy')
plt.plot(Validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.savefig('accuracy_and_validation_accuracy.png')
plt.show()

"""
plot the validation loss and loss graph
"""

plt.figure(figsize=(8,8))
plt.plot( loss, label='Training Loss')
plt.plot( validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Training_and_validation_loss.png')
plt.show()