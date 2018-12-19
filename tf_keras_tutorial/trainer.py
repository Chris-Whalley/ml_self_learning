# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:03:50 2018

@author: ckwha
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def train_cnn(train_images, train_labels, epochs):
	model = keras.Sequential()

	model.add(keras.layers.Conv2D(32, kernel_size = 3, strides=2, activation='relu', input_shape=[28, 28, 1]))
	model.add(keras.layers.MaxPool2D())
	model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'))
	model.add(keras.layers.MaxPool2D())
	model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(512, activation='relu'))
	model.add(keras.layers.Dense(10, activation='softmax'))

	model.compile(optimizer=keras.optimizers.Adam(),
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs=epochs, batch_size=64, verbose=1)
	model.summary()
	print('\n')

	return model

def train_deep(train_images, train_labels, epochs):
	model = keras.Sequential()

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(32, activation='relu'))
	model.add(keras.layers.Dense(10, activation='softmax'))

	model.compile(optimizer=keras.optimizers.Adam(),
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs=epochs, batch_size=64, verbose=1)
	model.summary()
	print('\n')

	return model