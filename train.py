import json
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


DATA_PATH= "data.json"
SAVED_MODEL_PATH="model.h5"

LEARNING_RATE=0.0001
EPOCHS=30 #40 kez deneniyor.
BATCH_SIZE=32

NUM_KEYWORDS=10 #up,down vs gibi 10tane ekledik ya .json da ilk başta mappings teki kelimeler adedince.

def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data=json.load(fp)

    #extract inputs and targets
    X=np.array(data["MFCCs"])
    y=np.array(data["labels"])

    return X,y

def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    #load dataset
    X,y= load_dataset(data_path)

    #create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
    X_train,X_validation, y_train, y_validation = train_test_split(X_train,y_train, test_size=test_validation)

    #convert inputs from 2d to 3d arrays
    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]
    X_test=X_test[...,np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

    #build_network
    model=keras.Sequential()

    #conv layer1
    model.add(keras.layers.Conv2D(64,(3,3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))

    #conv layer2
    model.add(keras.layers.Conv2D(32,(3,3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))

    #conv layer3
    model.add(keras.layers.Conv2D(32,(2,2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding="same"))

    #flatten the output feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    #softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS,activation="softmax")) #[0.1, 0.7, 0.2] gibi tahminler kullanır.

    #compile the model
    #optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    optimiser = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    #prit model overview
    model.summary()

    return model


def main():
    #load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    #build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) #(#segments, #cofficients 13, 1)
    #Mfcc de segmentleme vardı, 13 katsayısı kullanmıştık ve derinlik gri resimde 1, rgb görüntü olsaydı 3 olurdu.
    model=build_model(input_shape, LEARNING_RATE)


    #train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    #evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    #save the model
    model.save(SAVED_MODEL_PATH)

if __name__=="__main__":
    main()