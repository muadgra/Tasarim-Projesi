# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import time
NAME= f"model{int(time.time())}"
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

writer = tf.summary.create_file_writer("logs/graph_vis")

def my_func(x, y):
    return model



DATASET_PATH = "data.json"
SAVE_PATH = "model.h5"
EPOCHS = 30
BATCH_SIZE = 32
NUM_OF_CLASSES = 31
lr = 0.0001

def load_dataset(path):
    with open(path, "r") as file:
        data = json.load(file)
    
    return np.array(data["MFCCs"]), np.array(data["labels"])

def get_split_data(path, test_size = 0.1, val_size = 0.1):
    
    X, y = load_dataset(path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size)
    
    #add new dimension for CNN (number of segments, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    model.add(tf.keras.layers.Dense(NUM_OF_CLASSES, activation='softmax'))

    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    model.summary()
    
    return model
    
    
if __name__ == "__main__":
    log_dir = "./logs/"
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(DATASET_PATH)
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(X_train.shape)
    model = build_model(input_shape)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
   
        
    model.fit(X_train, y_train, epochs = EPOCHS, 
              batch_size = BATCH_SIZE,
              validation_data = (X_val, y_val),
              callbacks=[tensorboard])
    
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    
    print(f"Test error : {test_error} , Test_accuracy: {test_accuracy}")
    model.save(SAVE_PATH)
    
    
    
    
    
    
    
    
    