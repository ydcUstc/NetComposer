import os


#local path
DATA_DIR_LENGTH=len("..\_Store\data/")
DATA_DIR="..\_Store\data"
# Move file save location
OUT_DIR = '..\_Store\output'
CODE_DIR='./'
'''
#server path
DATA_DIR_LENGTH=len("/ghome/yindc/DeepJ/data/")
DATA_DIR="/ghome/yindc/DeepJ/data"
# Move file save location
OUT_DIR = '/ghome/yindc/DeepJ/output'
CODE_DIR='/ghome/yindc/DeepJ'
'''

# Define the musical styles
genre = [
    'baroque',
    'classical',
    'romantic'
]

styles = [
    [
        os.path.join(DATA_DIR, 'baroque/bach')
        #'data/baroque/handel',
        #'data/baroque/pachelbel'
    ],
    [
        os.path.join(DATA_DIR, 'classical/burgmueller'),
        #'data/classical/clementi',
        os.path.join(DATA_DIR, 'classical/haydn'),
        os.path.join(DATA_DIR, 'classical/beethoven'),
        os.path.join(DATA_DIR,'classical/brahms' ),
        os.path.join(DATA_DIR,'classical/mozart' )
        #'data/classical/mozart'
    ],
    [
        os.path.join(DATA_DIR, 'romantic/balakirew'),
        os.path.join(DATA_DIR, 'romantic/borodin'),
        os.path.join(DATA_DIR, 'romantic/brahms'),
        os.path.join(DATA_DIR, 'romantic/chopin'),
        os.path.join(DATA_DIR, 'romantic/debussy'),
        os.path.join(DATA_DIR, 'romantic/liszt'),
        os.path.join(DATA_DIR, 'romantic/mendelssohn'),
        #'data/romantic/moszkowski',
        os.path.join(DATA_DIR, 'romantic/mussorgsky'),
        #'data/romantic/rachmaninov',
        os.path.join(DATA_DIR, 'romantic/schubert'),
        os.path.join(DATA_DIR, 'romantic/schumann'),
        #'data/romantic/tchaikovsky',
        os.path.join(DATA_DIR, 'romantic/tschai')
    ]
]

NUM_STYLES = sum(len(s) for s in styles)

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 12

# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 8 * NOTES_PER_BAR

# Hyper Parameters
OCTAVE_UNITS = 64
STYLE_UNITS = 64
NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2


##OUT_DIR = './output'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')

