#!/usr/local/bin/python
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D, Activation
from keras.layers import Embedding, Flatten
from keras.layers.merge import Concatenate, Add
from keras.models import Model
import keras.backend as K
from keras import losses

from util import *
from constants import *
from keras.utils import plot_model
import pydotplus as pydot


#pydot.Dot.create(pydot.Dot())
model=Sequential()
model.add(Dense(8,input_shape=(16,)))
print("ok")
plot_model(model, to_file='model.png')