#!/usr/bin/env py thon
# coding: utf-8
from dnn import L_layer_DNN as nn 
import numpy as np
def load_datas():
    import h5py
    train_dataset = h5py.File('/home/sravanm/DS_ML/build_nn-from-scratch/utils/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('/home/sravanm/DS_ML/build_nn-from-scratch/utils/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_datas()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255
test_x = test_x_flatten/255

# 2 Layers with relu->sigmoid activations
layer_dims = [train_x.shape[0], 7, 1]
activations = ['linear', 'relu', 'sigmoid']

model = nn(layer_dims, activations)
model.fit(train_x, train_y)

A = model.predict(train_x)

model.scores(train_y[0], A[0])
