#!
import numpy as np
import tensorflow as tf
from collections import deque
import midi
import argparse
import scipy.io as scio

from constants import *
from util import *
from dataset import *
from tqdm import tqdm
from midi_util import midi_encode
from midi_util import update_played, get_volume

class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, style, default_temp=1, tone='Grand_piano', classes=MIDI_MAX_NOTES):
        self.input_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        #self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

        # The next note being built
        self.next_output = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.next_input = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

        dataFile = os.path.join(CODE_DIR,'tone_'+tone+'.mat')
        data = scio.loadmat(dataFile)
        self.tone = data['tone']

        #instrument state parameters
        self.instrument_classes=classes
        #record time after notes is played
        self.played_steps=np.zeros(classes)
        #record replay velocities
        self.velocities=np.zeros(classes)
        #record stop note events
        self.stop_notes=np.zeros(classes)

    def build_time_inputs(self):
        return (
            np.array(self.input_memory),
            #np.array(self.beat_memory),
            np.array(self.style_memory)
        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_input]),
            np.array(list(self.style_memory)[-1:])
        )

    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)

        # Stop notes randomly
        if np.random.random() <= prob[1] or self.played_steps[n + MIN_NOTE]==80:
            #output's stop note event
            self.next_output[n, 1] = 1
            #update instrument's states
            self.played_steps[n + MIN_NOTE] = -1
            self.velocities[n + MIN_NOTE] = 0
            self.stop_notes[n + MIN_NOTE] = 1
        # Flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_output[n, 0] = 1
            # Apply volume
            self.next_output[n, 2] = vol
            # update instrument's states
            self.velocities[n + MIN_NOTE] = vol
            self.played_steps[n + MIN_NOTE] = 1
            if vol == 0:
                self.played_steps[n + MIN_NOTE] = -1
                self.stop_notes[n + MIN_NOTE] = 1


    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.played_steps) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        #adjust output and instrument's states to next input format
        self.next_input[:, 0] = self.stop_notes[MIN_NOTE:MAX_NOTE]
        self.next_input[:, 1] = self.next_output[:, 2]
        self.next_input[:, 2] = get_volume(self.velocities, self.played_steps, self.tone)[MIN_NOTE:MAX_NOTE]

        self.input_memory.append(self.next_input)
        # Consistent with dataset representation
        #self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_output)
        # Reset next note (alloc new arrays)
        self.next_input = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.next_output = np.zeros((NUM_NOTES, NOTE_UNITS))
        #update instrument's states
        self.played_steps=update_played(self.played_steps)
        self.stop_notes = np.zeros(self.instrument_classes)
        return self.results[-1]

def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / prob - 1)
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars, styles):
    print('Generating with styles:', styles)

    _, time_model, note_model = models
    generations = [MusicGeneration(style) for style in styles]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        # Pick only the last time step
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            predictions = np.array(note_model.predict(ins))

            for i, g in enumerate(generations):
                # Remove the temporal dimension
                # choose note n's predictions in the i th style, according to the probability based on the last note
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    ##np.save(SAMPLES_DIR, results)
    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '_' + str(i) + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = midi_encode(unclamp_midi(result))

        midi.write_midifile(fpath, mf)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--bars', default=32, type=int, help='Number of bars to generate')
    parser.add_argument('--styles', default=None, type=int, nargs='+', help='Styles to mix together')
    args = parser.parse_args()

    models = build_or_load()

    styles = [compute_genre(i) for i in range(len(genre))]

    if args.styles:
        # Custom style
        styles = [np.mean([one_hot(i, NUM_STYLES) for i in args.styles], axis=0)]

    #write_file('output', generate(models, args.bars, styles))
    generated=generate(models, 20, styles)
    write_file('output', generated)

if __name__ == '__main__':
    main()
