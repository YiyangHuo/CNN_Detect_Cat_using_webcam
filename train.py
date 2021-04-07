#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 4/6/21 11:08 PM
#@Author: Yiyang Huo
#@File  : train.py.py



from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
from keras.utils import to_categorical
import os

IMAGE_SIZE = 256
IMAGE_TRAIN_DIRECTORY = './data/training_set'
IMAGE_TEST_DIRECTORY = './data/test_set'


def label_img(name):
    if name == 'cats': return np.array([1, 0])
    elif name == 'notcats' : return np.array([0, 1])

def load_data(directory):
    print("Loading images...")
    train_data = []
    directories = next(os.walk(directory))[1]

    for dirname in directories:
        print("Loading {0}".format(dirname))
        file_names = next(os.walk(os.path.join(directory, dirname)))[2]
        for i in range(500):
            image_name = choice(file_names)
            image_path = os.path.join(directory, dirname, image_name)
            label = label_img(dirname)
            if "DS_Store" not in image_path:
                img = Image.open(image_path)
                img = img.convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                arrayimg = np.array(img)
                train_data.append([arrayimg, label])

    return train_data

def training_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model

    return model



if __name__ == "__main__":
    training_data = load_data(IMAGE_TRAIN_DIRECTORY)
    model = training_model()
    training_images = np.array([i[0] for i in training_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    training_labels = np.array([i[1] for i in training_data])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_images, training_labels, batch_size=50, epochs=10, verbose=1)
    model.save("model2.h5")