import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
import argparse
import midi
import os

from constants import *
from dataset import *
from generate import *
from midi_util import midi_encode
from model import *

train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN);