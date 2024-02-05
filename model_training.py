import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
import cv2
import numpy as np
from tensorflow.keras import regularizers
test=tf.keras.utils.image_dataset_from_directory(
    r"/content/Untitled Folder 1/dataset/validation",
    batch_size=32,
    image_size=(64,64)
)
test= test.map(lambda x, y: (x / 255, y))
train=tf.keras.utils.image_dataset_from_directory(
    r"/content/Untitled Folder 1/dataset/train",
    batch_size=32,
    image_size=(64,64))

train= train.map(lambda x, y: (x / 255, y))




md = Sequential()
md.add(Conv2D(filters=20, kernel_size=(3, 3),activation="relu", input_shape=(64,64, 3)))
md.add(MaxPooling2D(2,2))
md.add(Conv2D(filters=10, kernel_size=(2, 2), activation="relu"))
md.add(MaxPooling2D(2,2))
md.add(Flatten())
md.add(Dense(256,activation='relu'))
md.add(layers.Dropout(0.5))
md.add(Dense(128,activation='relu'))
md.add(layers.Dropout(0.5))
md.add(Dense(64,activation='relu'))
md.add(layers.Dropout(0.5))
md.add(Dense(26, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

md.compile(optimizer='rmsprop',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
md.fit(train, epochs=20, validation_data=test, validation_steps=len(test), callbacks=[early_stopping])

md.save(r"/content/Untitled Folder 2")