#!/usr/bin/env python
# coding: utf-8

# In[2]:


#For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input

#########################################################



import tensorflow as tf
# Import PySwarms
import pyswarms as ps



# Import modules
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
import matplotlib

###########################################################

from keras.utils import to_categorical
data_train = pd.read_csv('fashion-mnist_train.csv')
data_test = pd.read_csv('fashion-mnist_test.csv')

X_train = np.array(data_train.iloc[:, 1:])
X_test = np.array(data_test.iloc[:, 1:])
y_train = to_categorical(np.array(data_train.iloc[:, 0]))
y_test = to_categorical(np.array(data_test.iloc[:, 0]))


##########################################################

img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Store the features as X and the labels as y
X = X_train
y = y_train


# In[2]:


# Convolutional Neural Network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf 
visible = Input(shape=(28,28,1))
conv1 = Conv2D(32, kernel_size=3, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(64, kernel_size=3, activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#conv4 = Conv2D(16, kernel_size=4, activation='relu')(pool3)
#pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#placeHolder1 = tf.placeholder(tf.float32,shape=(None, 800)) 
flat = Flatten()(pool3) 

sess = tf.Session()

#<class 'tensorflow.python.framework.ops.Tensor'>
tf.InteractiveSession()  # run an interactive session in Tf.
 
flat_np= tf.stack([flat])
X=flat_np
#flat_np = flat.eval()
print(type(flat_np))



# In[3]:


# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss. It receives a set of parameters that must be
    rolled-back into the corresponding weights and biases.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """
    # Neural network architecture
    n_inputs = 1600
    n_hidden = 2000
    #n_hidden_2 = 1000
    n_classes = 10

    # Roll-back the weights and biases
    W1 = params[0:3200000].reshape((n_inputs,n_hidden))
    b1 = params[3200000:3202000].reshape((n_hidden,))
    W2 = params[3202000:3222000].reshape((n_hidden,n_classes,))
    b2 = params[3222000:3222010].reshape((n_classes,))

    #X=flat_np

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)    # np.exp means exponential function
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   #axis=1 means columns

    # Compute for the negative log likelihood
    N = 150 # Number of samples
    corect_logprobs = -np.log(probs[range(N), y])
    loss = np.sum(corect_logprobs) / N

    return loss


# In[4]:


def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(X[i]) for i in range(n_particles)]
    return np.array(j)


# In[5]:


import pyswarms as ps

from pyswarms.utils.functions import single_obj as fx

get_ipython().run_line_magic('time', '')

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
dimensions = ( 1600 * 2000) + (2000 * 10) + 2000 + 10
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)


# In[2]:


def predict(X, pos):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    
    # Neural network architecture
    n_inputs = 1600
    n_hidden = 2000
    #n_hidden_2 = 1000
    n_classes = 10

    # Roll-back the weights and biases
    W1 = params[0:3200000].reshape((n_inputs,n_hidden))
    b1 = params[3200000:3202000].reshape((n_hidden,))
    W2 = params[3202000:3222000].reshape((n_hidden,n_classes,))
    b2 = params[3222000:3222010].reshape((n_classes,))



    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred


# In[8]:


import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical
data_train = pd.read_csv('fashion-mnist_train.csv')
data_test = pd.read_csv('fashion-mnist_test.csv')

X_train = np.array(data_train.iloc[:, 1:])
X_test = np.array(data_test.iloc[:, 1:])
y_train = to_categorical(np.array(data_train.iloc[:, 0]))
y_test = to_categorical(np.array(data_test.iloc[:, 0]))


##########################################################

img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Store the features as X and the labels as y
X = X_train
y = y_train


##########################################################

X= X_test
y= y_test
(predict(X, pos) == y).mean()


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




