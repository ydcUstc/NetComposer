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
import scipy.io as scio

models = build_or_load()
train_data, train_labels = load_part(styles, BATCH_SIZE, SEQ_LEN,load_probability=0.5)
'''
midireader=midi.FileReader()
midifile = open("./data/baroque/bach1/bach_846.mid", 'rb')
pattern=midireader.read(midifile)
fpath = "./testio.mid"
print('Writing file', fpath)
os.makedirs(os.path.dirname(fpath), exist_ok=True)
midi.write_midifile(fpath,pattern)

#print(bytes(chr(192),encoding="utf-8"))
dataFile = './tone_Grand_piano.mat'
data = scio.loadmat(dataFile)
a=data['tone']
print("loaded")

a=[1,2,2,4,5,5,6]
print(a.index(2))

style=os.path.join(DATA_DIR, 'classical/burgmueller')
files=get_all_files([style])
print("got file paths:"+files[1])
seq=load_midi(files[1])
note_seq=seq[:,:,:2]
print("loaded note sequence")
view_data=np.swapaxes(seq,0,2)
pattern=midi_encode(note_seq, resolution=10, step=1)
print("created pattern")
fpath = "./testio.mid"
print('Writing file', fpath)
os.makedirs(os.path.dirname(fpath), exist_ok=True)
midi.write_midifile(fpath,pattern)'''