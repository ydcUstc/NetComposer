####!/usr/local/bin/python
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

def main():
    models = build_or_load()
    train(models)

def train(models):
    print('Loading data')
    train_data, train_labels = load_part(styles, BATCH_SIZE, SEQ_LEN, load_probability=0.3)
    test_data, test_labels =load_part(styles, BATCH_SIZE, SEQ_LEN, load_probability=0.3)

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir=os.path.join(OUT_DIR,'logs'), histogram_freq=1)
    ]

    print('Training')
    models[0].fit(train_data, train_labels, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE,validation_data=(test_data, test_labels))

if __name__ == '__main__':
    main()
