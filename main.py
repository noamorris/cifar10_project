{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "NTwNZcxOOU6W"
   },
   "outputs": [],
   "source": [import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
             ]]}



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()



train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.2)
train_iter = datagen.flow(train_images, train_labels, batch_size=64)



model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()



history = model.fit(train_iter, epochs=10, validation_data=(test_images, test_labels))



yhat = model.predict(test_images)

fig = plt.figure()
fig.set_size_inches(20, 20)
graphIndex = 0

classifiers = {
    0: 'airplane', 
    1: 'automobile', 
    2: 'bird', 
    3: 'cat',
    4: 'deer',
    5: 'dog', 
    6: 'frog', 
    7: 'horse', 
    8: 'ship', 
    9: 'truck'
}

for i in range(random.randrange(0, len(yhat))):   # Choose a random starting point in the prediction list
  realIndex = np.where(test_labels[i] == max(test_labels[i]))   # Get the the number that the NN is supposed to predict
  predIndex = np.where(yhat[i] == max(yhat[i]))       # Get the number that the NN did predict
  if predIndex == realIndex:    # If the prediction is not right, graph it
    graphIndex += 1
    try:
      plt.subplot(5, 5, graphIndex)
    except ValueError:    # After 25 plots then stop the loop
      break
    plt.imshow(test_images[i].reshape(32, 32, 3))
    plt.title(f"Real: {classifiers[realIndex[0][0]]}\nPredicted: {classifiers[predIndex[0][0]]}", fontdict={'fontsize': 15})
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

plt.axis('off')
plt.tight_layout()
plt.show()



classifiers = {
    0: 'airplane', 
    1: 'automobile', 
    2: 'bird', 
    3: 'cat',
    4: 'deer',
    5: 'dog', 
    6: 'frog', 
    7: 'horse', 
    8: 'ship', 
    9: 'truck'
}

# class: [total, correct]
total = {
    'airplane': [0, 0],
    'automobile': [0, 0],
    'bird': [0, 0],
    'cat': [0, 0],
    'deer': [0, 0],
    'dog': [0, 0],
    'frog': [0, 0],
    'horse': [0, 0],
    'ship': [0, 0],
    'truck': [0, 0]
}

accuracy = {
    'airplane': 0,
    'automobile': 0,
    'bird': 0,
    'cat': 0,
    'deer': 0,
    'dog': 0,
    'frog': 0,
    'horse': 0,
    'ship': 0,
    'truck': 0
}

yhat = model.predict(test_images)
for predRow, realRow in zip(yhat, test_labels):
  predIndex = np.argmax(predRow)
  realIndex = np.argmax(realRow)
  total[classifiers[predIndex]][0] += 1 # Add 1 to the total for the respective class
  
  if predIndex == realIndex:
    total[classifiers[predIndex]][1] += 1 # Add 1 correct guess for the respective class

for c, l in total.items():
  accuracy[c] = l[1] / l[0] * 100

accuracy = {k: v for k, v in sorted(accuracy.items(), key=lambda item: item[1])} # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
plt.barh(list(accuracy.keys()), list(accuracy.values()))
plt.xticks(ticks=np.arange(0, 100, 5))
plt.show()
