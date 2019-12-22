import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
import h5py
import argparse
import matplotlib.pyplot as plt
import keras.backend as K
K.set_image_data_format('channels_last')

# Returns the dataset
def loadDataset():
    train_dataset = h5py.File('data/train_happy.h5', "r")
    train_x = np.array(train_dataset["train_set_x"])/255.
    train_y = np.array(train_dataset["train_set_y"])

    test_dataset = h5py.File('data/test_happy.h5', "r")
    test_x = np.array(test_dataset["test_set_x"])/255.
    test_y = np.array(test_dataset["test_set_y"])

    classes = np.array(test_dataset["list_classes"])
    
    train_y = train_y.reshape((1, train_y.shape[0])).T
    test_y = test_y.reshape((1, test_y.shape[0])).T
    
    return [train_x, train_y, test_x, test_y, classes]

# Displays an image from the dataset
def displayImage(img):
    plt.imshow(img.reshape(64,64,3))
    plt.show()


def MyModel(in_shape):
    X_input = Input(in_shape)

    X = ZeroPadding2D((3,3))(X_input)

    # Conv 1 layer
    X = Conv2D(32, (7,7), strides = (1,1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # Max pooling
    X = MaxPooling2D((2,2), name = 'max-pool')(X)

    # Flatten for the fully connected layer
    X = Flatten()(X)
    X = Dense(1, activation = 'sigmoid', name = 'fc')(X)

    # create a model and return the same
    model = Model(inputs = X_input, outputs = X, name = 'MyModel')

    return model