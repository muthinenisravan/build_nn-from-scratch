#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
sys.path.insert(1, '/home/sravanm/DS_ML/build_nn-from-scratch/utils')
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, relu,sigmoid_backward, relu_backward,load_data, compute_cost
from copy import deepcopy
np.random.seed(1)

class L_layer_DNN:

    def __init__(self, l_dims, l_activations, learning_rate = 0.0075, iterations = 2500):
        self.l_dims = l_dims
        self.l_activations = l_activations
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.L = len(l_activations)-1
        self.parameters = self.initialize_parameters_deep()
#        print (self.parameters)

    def accuracy(self,y,y_pred):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y,y_pred)

    def scores(self, y, y_pred):
        from sklearn.metrics import classification_report
        print (classification_report(y,y_pred))


    def initialize_parameters_deep(self):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
    
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
    
        np.random.seed(1)
        layer_dims = self.l_dims
        acts = self.l_activations
        parameters = {}
    
        for l in range(1, self.L+1):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            parameters["act"+str(l)] = acts[l]
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        
        return parameters

    def _compute_fw_activation_(self, z, activation):

        if(activation == 'sigmoid'):
            return  sigmoid(z)
        elif (activation == 'relu'):
            return  relu(z)
        elif (activation == 'tanh'):
            return  tanh(z)
        else: 
            pass

        return z,'linear'

    def _compute_fw_linear_(self, l_input, w, b):
    
        z = np.dot(w,l_input)+b
        l_cache = (l_input, w, b)
    
        return z, l_cache

    def _compute_linear_activation_farward_(self, A_in, w, b, activation):

        z,l_cache = self._compute_fw_linear_(A_in, w, b)
        A_out, a_cache = self._compute_fw_activation_(z, activation)
        c = (l_cache, a_cache)

        return A_out, c

    def compute_forward_propagation(self, in_x):
        params = self.parameters

    #params - > W,b & Activation details of layer
        A = in_x
   
        caches = list()
 
        for i in range(1,self.L+1):
            w = params['W'+str(i)]
            b = params['b'+str(i)]
            activation = params['act'+str(i)]
            A_prev = A
            A, cache = self._compute_linear_activation_farward_(A_prev, w, b, activation)
            caches.append(cache)
    
        return A, caches

    def _compute_activation_bw_(self, dA, cache, activation):
    
        if( activation == 'sigmoid'):
            return sigmoid_backward(dA, cache)
        elif (activation == 'relu'):
            return relu_backward(dA, cache)
        else:
            return dA

    def _compute_linear_bw_(self, dZ, cache):
        """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T)/m
        db = np.sum(dZ,axis=1,keepdims=1)/m
        dA_prev = np.dot(W.T, dZ)
    
        return dA_prev, dW, db



    def _compute_linear_activation_backward_(self, dA, cache):
    
        # l_cache -> (A_prev, W,b)
        # a_cache -> (activation, Z)
        l_cache, a_cache = cache
        activation , z = a_cache 
    
        dZ = self._compute_activation_bw_(dA, z, activation)
        dA_p, dW, db = self._compute_linear_bw_(dZ, l_cache)
 
        return dA_p, dW, db

    def compute_backward_propagation(self, Y):
    
        # Making sure Fw calculation matches with Output
        #AL-> post activation value of Final layer
        AL = self.AL
        caches = self.caches
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
    
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        grads = dict()
    
        dA = dAL
    
        for i in reversed(range(self.L)):
            cache = caches[i]
            dA, dW, db = self._compute_linear_activation_backward_(dA, cache)

            grads['dA'+str(i)] = dA
            grads['dW'+str(i+1)] = dW
            grads['db'+str(i+1)] = db
        
        return grads



    def update_parameters(self):

        # Each Param-> W, b, Activations * L params
        parameters = deepcopy(self.parameters)
    
        for l in range(self.L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] -  self.learning_rate*self.grads["dW" + str(l+1)] 
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] -  self.learning_rate*self.grads["db" + str(l+1)] 
        return parameters



    def predict(self, X):
        A, caches = self.compute_forward_propagation(X)
        A = A>0.5
        return A

    def fit(self, X, Y, print_cost = False):
        for i in range(self.iterations):
            self.AL, self.caches  = self.compute_forward_propagation(X)
            self.grads = self.compute_backward_propagation(Y)
            self.parameters = self.update_parameters()
            cost = compute_cost(Y, self.AL)
            if((print_cost and (i%50==0)) or (i == self.iterations-1) ):
                print ("Cost at:"+str(i)+" is "+str(cost))
