"""
In this project, we use the Sequential deep learning model for potato disease classification
There are three classes in this.
Here is the link where you can download the dataset ( https://www.kaggle.com/datasets/arjuntejaswi/plant-village )
"""

"""
import libraraies 
"""
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# define the image size, number of channels, epochs, and batch size
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=5

# Load the dataset using tensorflow from computer directory
dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "potato_disease",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
    )

# Print the class names 
class_names=dataset.class_names
print(class_names)

# print the information about the first batch (image_shape, batch_size, labels) 
for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

# show the samples image from the dataset
plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


"""
The below function is used to split the dataset into
training, test, and validation dataset
"""
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    data_size=len(dataset)
    
    if shuffle:
        ds=ds.shuffle(shuffle_size, seed=12)
    
    train_size=int(train_split*data_size)
    val_size=int(val_split*data_size)
    
    train_data=ds.take(train_size)
    val_data=ds.skip(train_size).take(val_size)
    test_data=ds.skip(train_size).skip(val_size)
    return train_data, val_data, test_data

"""
here we call the split function 
"""    
train_data, val_data, test_data=get_dataset_partitions_tf(dataset)


"""
Write the preprocessing layers for resizing the image 
and rescaling it in between 0 and 1
"""
resizing_and_rescaling=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
    ])

"""
Data augmentation layers also added in our model to
make our model robust
"""
data_augmentation_layer=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
    ])
 
"""
model building
"""

input_size=(BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
classes=3

model=models.Sequential([
    resizing_and_rescaling,
    data_augmentation_layer,
    layers.Conv2D(32,(3*3),activation='relu', input_shape=input_size),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3*3),activation='relu', padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3*3),activation='relu', padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3*3),activation='relu', padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3*3),activation='relu', padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3*3),activation='relu', padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(classes, activation='softmax')
    ])

model.build(input_shape=input_size)

"""
Print the model Summary
"""

model.summary()

"""
compile the model with well-known optimizer adam
Sparse Categorical cross-entropy loss is used because 
of multi-class classification
"""

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
    )

"""
model trainig on train dataset which is 80% of the 
dataset
"""

history=model.fit(
    train_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=val_data
    )

"""
Evaluate the model's accuracy and loss on test dataset
"""

Score=model.evaluate(test_data)
print('Accuracy on test dataset')
print(Score)

accuracy=history.history['accuracy']
loss=history.history['loss']
validation_loss=history.history['val_loss']
Validation_accuracy=history.history['val_accuracy']

"""
plot the accuracy and validation accuracy graph
"""
plt.figure(figsize=(8,8))
plt.plot(range(EPOCHS), accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), Validation_accuracy, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')
plt.show()

"""
plot the validation and Training loss graph
"""

plt.figure(figsize=(8,8))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
