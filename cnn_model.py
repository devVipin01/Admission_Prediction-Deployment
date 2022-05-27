import os
import numpy as np
'''
import cv2
from glob import glob
import itertools
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix

# TensorFlow Modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#train_dir='E:\\image_data\\train'
#test_dir='E:\\image_data\\test'
#target = ['Abhishek', 'Vipin']

def data_processing(train_dir,test_dir):
    #this function return train & test data

    global IMG_SHAPE
    train_datagenerator = ImageDataGenerator(rescale=1./255,
                                         rotation_range=20,
                                         width_shift_range=.15,
                                         height_shift_range=.15,
                                         horizontal_flip=True,
                                         zoom_range=0.2,
                                         data_format = "channels_last")

    test_datagenerator = ImageDataGenerator(rescale = 1./255)
    
    batch_size = 64 # number of images to process at a time 
    IMG_SHAPE = 224 # convert all images to be 224 x 224
    train_data = train_datagenerator.flow_from_directory(directory = train_dir,
                                                     batch_size = batch_size,
                                                     target_size = (IMG_SHAPE, IMG_SHAPE),
                                                     shuffle = True, #for training only
                                                     class_mode = 'sparse', #type of problem (sparse, binary, or categorical, etc.)
                                                     classes = target)  

    test_data = test_datagenerator.flow_from_directory(directory = test_dir,
                                                       batch_size = batch_size,
                                                       target_size = (IMG_SHAPE, IMG_SHAPE),
                                                       shuffle = False,
                                                       class_mode = 'sparse',
                                                       classes = target)
    return train_data,test_data


def train_model(train_data,test_data,target):
    global model
    loss = "sparse_categorical_crossentropy" 
    output_units = len(target)
    output_activation = 'softmax'
    learning_rate = 0.0005 #Default = 0.001
    epochs = 12
    print(epochs)
    NUM_COLOR_CHANNELS = 3
    model = tf.keras.models.Sequential([
        Input(shape = (IMG_SHAPE, IMG_SHAPE, NUM_COLOR_CHANNELS)),
        Conv2D(filters=16, kernel_size=3, activation="relu"),
        Conv2D(32, 3, activation="relu"),
        MaxPool2D(pool_size=2, padding="valid"), # padding can also be 'same'
        Conv2D(64, 3, activation="relu"),
        Conv2D(128, 3, activation="relu"),
        GlobalMaxPool2D(),
        Dense(64, activation = 'relu'),
        Dropout(0.2),
        Dense(output_units, activation=output_activation) ])
    model.compile(loss=loss,optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])
# model summary
#model.summary()
# Fit the model
    history = model.fit(train_data,epochs=epochs,validation_data=test_data)
    loss, accuracy = model.evaluate(test_data)
    return model,history,loss,accuracy

def cnn_driver(train_dir,test_dir,target):
    global test_data
    train_data,test_data=data_processing(train_dir,test_dir)
    model,history,loss,accuracy=train_model(train_data,test_data,target)
    print("all donne")
    
    return model,history,loss,accuracy
    '''