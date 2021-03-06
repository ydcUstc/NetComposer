"""
Preprocesses MIDI files
"""
import sys
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *
from midi_util import load_midi
from util import *

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / len_melody])

def compute_genre(genre_id):
    """ Computes a vector that represents a particular genre """
    genre_hot = np.zeros((NUM_STYLES,))
    start_index = sum(len(s) for i, s in enumerate(styles) if i < genre_id)
    styles_in_genre = len(styles[genre_id])
    genre_hot[start_index:start_index + styles_in_genre] = 1 / styles_in_genre
    return genre_hot

def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first event
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)

    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def load_all(styles, batch_size, time_steps):
    """
    Loads all MIDI files as a piano roll.
    (For Keras)
    """
    note_data = []
    beat_data = []
    style_data = []

    note_target = []

    # TODO: Can speed this up with better parallel loading. Order gaurentee.
    styles = [y for x in styles for y in x]

    for style_id, style in enumerate(styles):
        style_hot = one_hot(style_id, NUM_STYLES)
        # Parallel process all files into a list of music sequences
        #for f in get_all_files([style]):
            #print("loading"+f)
        seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(f) for f in get_all_files([style]))

        for seq_id,seq in enumerate(seqs):
            f = get_all_files([style])[seq_id]
            if len(seq) >= time_steps:
                # Clamp MIDI to note range
                seq = clamp_midi(seq)
                # Create training data and labels
                train_data, _ = stagger(seq, time_steps)
                _, label_data = stagger(seq[:][:][:1], time_steps)
                note_data += train_data
                note_target += label_data
                #view_data=np.swapaxes(seq,0,2)


                style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

    note_data = np.array(note_data)
    style_data = np.array(style_data)
    note_target = np.array(note_target)
    return [note_data, note_target, style_data], [note_target]

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')


def load_part(styles, batch_size, time_steps, load_probability):
    """
    Loads part of all MIDI files as a piano roll randomly.
    Used when there is not enough space for all MIDI
    (For Keras)
    """
    note_data = []
    beat_data = []
    style_data = []

    note_target = []

    # TODO: Can speed this up with better parallel loading. Order gaurentee.
    styles = [y for x in styles for y in x]

    for style_id, style in enumerate(styles):
        files=get_all_files([style])
        is_loaded=np.random.rand(len(files))
        style_hot = one_hot(style_id, NUM_STYLES)
        # Parallel process all files into a list of music sequences
        #for f in get_all_files([style]):
            #print("loading"+f)
        seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(f) for f in files)

        for seq_id,seq in enumerate(seqs):
            f = get_all_files([style])[seq_id]
            if is_loaded[seq_id]<load_probability:
                print("appending "+f+" as dataset")
                sys.stdout.flush()
                if len(seq) >= time_steps:
                    # Clamp MIDI to note range
                    seq = clamp_midi(seq)
                    # Create training data and labels
                    train_data, _ = stagger(seq[:,:,1:], time_steps)
                    _, label_data = stagger(seq[:,:,:3], time_steps)
                    note_data += train_data
                    note_target += label_data
                    view_data=np.swapaxes(train_data,1,3)


                    style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]


    note_data = np.array(note_data)
    style_data = np.array(style_data)
    note_target = np.array(note_target)
    return [note_data, note_target, style_data], [note_target]