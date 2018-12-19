# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:07:23 2018

@author: ckwha
"""
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import plotter
import trainer

#print(tf.__version__)

# =============================================================================
# User input
# =============================================================================
mode = input("Train network? [y or n]\n")
dataset = input("Which dataset? [1 or 2]\n")

if mode == 'y' or mode == 'Y':
	network = input("Which network? [cnn or deep]\n")
	epochs = int(input("How many epochs? [integer]\n"))

# =============================================================================
# Data import
# =============================================================================
mnist = keras.datasets.mnist
fashion_mnist = keras.datasets.fashion_mnist

if dataset == '1':
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
elif dataset == '2':
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_originals = test_images
train_images = np.reshape(train_images, (-1, 28, 28, 1))/ 255.0
test_images = np.reshape(test_images, (-1, 28, 28, 1))/ 255.0

# =============================================================================
# Model application
# =============================================================================
if mode == 'y' or mode == 'Y':
	if network == "cnn":
		model = trainer.train_cnn(train_images, train_labels, epochs)
	else:
		model = trainer.train_deep(train_images, train_labels, epochs)
	tf.keras.models.save_model(model, filepath = 'Model', overwrite=True, include_optimizer=True)
else:
	model = tf.keras.models.load_model(filepath = 'Model', custom_objects=None, compile=True)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

predictions = model.predict(test_images)

# =============================================================================
# Plotter
# =============================================================================
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 10
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plotter.plot_image(i, predictions, test_labels, test_originals,class_names)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plotter.plot_value_array(i, predictions, test_labels,class_names)