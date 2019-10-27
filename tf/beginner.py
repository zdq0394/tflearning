# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:03:30 2019

@author: zhangdongqi
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import common

(x_train, y_train), (x_test, y_test) = common.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test,  y_test, verbose=2)