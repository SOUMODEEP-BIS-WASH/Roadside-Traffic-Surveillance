# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 00:32:24 2023

@author: biswa
"""

import numpy as np
import itertools
from sklearn.metrics import accuracy_score

#IMPORTING LIBRARIES

import cv2
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
#READING AND PREPROCESSING OF IMAGE DATA

SIZE = 256
BATCH_SIZE=32

datagen= ImageDataGenerator(horizontal_flip=True,
                            rotation_range=45,
                            rescale=1.0 / 255)


train_generator = datagen.flow_from_directory('DATABASE/train/',target_size=(SIZE,SIZE),batch_size=BATCH_SIZE ,class_mode='categorical',shuffle=True,seed=123)
test_generator = datagen.flow_from_directory('DATABASE/final test/',target_size=(SIZE,SIZE),batch_size=BATCH_SIZE, class_mode='categorical',shuffle=False,seed=123)
true_labels = test_generator.classes

from keras.models import load_model
model1 = tf.keras.models.load_model('Accident_Detection_MobileNet.hdf5')
model2 = tf.keras.models.load_model('Accident_Detection_RESNET50.hdf5')
model3 = tf.keras.models.load_model('Accident_Detection_Xception.hdf5')
model4 = tf.keras.models.load_model('Accident_Detection_DENSENET201.hdf5')

model1_preds=model1.predict(test_generator)
model2_preds=model2.predict(test_generator)
model3_preds=model3.predict(test_generator)
model4_preds=model4.predict(test_generator)


weight_range = np.linspace(0, 1, num=11)
weight_combinations = list(itertools.product(weight_range, repeat=4))

best_accuracy = 0
best_weights = None

for w1, w2, w3,w4 in weight_combinations:
    ensemble_predictions = []

    for i in range(len(model1_preds)):
        weighted_average = (
            w1 * model1_preds[i] +
            w2 * model2_preds[i] +
            w3 * model3_preds[i]+
            w4*model4_preds[i]
        )
        final_prediction = np.argmax(weighted_average)
        ensemble_predictions.append(final_prediction)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, ensemble_predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = [w1, w2, w3,w4]

#best_accuracy_percentage = best_accuracy * 100

#print("Best Weights:", best_weights)
#print("Best Accuracy:", best_accuracy_percentage, "%")
# ... (Your code to find the best weights)

# Normalize the best weights
sum_best_weights = sum(best_weights)
normalized_best_weights = [w / sum_best_weights for w in best_weights]

#print("Normalized Best Weights:", normalized_best_weights)

# Now use the normalized best weights for the ensemble predictions
ensemble_predictions = []

for i in range(len(model1_preds)):
    weighted_average = (
        normalized_best_weights[0] * model1_preds[i] +
        normalized_best_weights[1] * model2_preds[i] +
        normalized_best_weights[2] * model3_preds[i]+
        normalized_best_weights[3]*model4_preds[i]
    )
    final_prediction = np.argmax(weighted_average)
    ensemble_predictions.append(final_prediction)

# Calculate accuracy
#ensemble_accuracy = accuracy_score(true_labels, ensemble_predictions)
#ensemble_accuracy_percentage = ensemble_accuracy * 100

#print("Ensemble Accuracy:", ensemble_accuracy_percentage, "%")

cam = cv2.VideoCapture(0)

def trigger(label):
    desired=['accident','fire']
    if label in desired:
        return True
    else:
        return False
    
while True:
    ret,frame=cam.read()
    
    if not ret:
        break
    
    selected_image = cv2.resize(frame, (SIZE, SIZE))
    selected_image_rgb = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    selected_image_normalized = selected_image_rgb / 255.0  # Normalize pixel values
    
    # Use the ensemble to predict the label
    weighted_average = (
    normalized_best_weights[0] * model1.predict(np.expand_dims(selected_image_normalized, axis=0)) +
    normalized_best_weights[1] * model2.predict(np.expand_dims(selected_image_normalized, axis=0)) +
    normalized_best_weights[2] * model3.predict(np.expand_dims(selected_image_normalized, axis=0)) +
    normalized_best_weights[3] * model4.predict(np.expand_dims(selected_image_normalized, axis=0))
    )
    ensemble_predicted_label = np.argmax(weighted_average)

    # Convert label indices to class labels using the generator
    class_labels = list(train_generator.class_indices.keys())
    ensemble_predicted_label_name = class_labels[ensemble_predicted_label]    
    
    if trigger(ensemble_predicted_label_name):
        print("Alert!!!")
        
    cv2.imshow('Camera View',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()


    