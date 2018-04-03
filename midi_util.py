"""
Handles MIDI file loading
"""
import sys
import midi

import numpy as np
import os
from constants import *
import scipy.io as scio

def get_replay(played_steps,velocities):
    replay = np.zeros(len(played_steps))
    for i in range(len(played_steps)):
        if played_steps[i] == 1:
            replay[i] = velocities[i]
    return replay

def get_stop(played_steps):
    stop=np.zeros(len(played_steps))
    for i in range(len(played_steps)):
        if played_steps[i] == -1:
            stop[i] = 1
    return stop

def update_played(played_steps):
    played= np.maximum(np.minimum(played_steps,1),0)
    stopped=np.minimum(played_steps,0)
    return np.around(played_steps+played-stopped)

def get_volume(velocities,played_steps,tone):
    classes=len(played_steps)
    arr=np.zeros(classes)
    for i in range(classes):
        a=int(round(played_steps[i]))
        if a<79 and a>=0:
            for stage in range(6):
                pitch=i+int(round(np.log2(stage+1)*12))
                if(pitch<classes):
                    arr[pitch] += tone[stage][a][i] * velocities[i]
    return arr

def midi_encode(note_seq, resolution=NOTES_PER_BEAT, step=1):
    """
    Takes a piano roll and encodes it into MIDI pattern
    """
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)

    stops = note_seq[:, :, 1]
    replays = note_seq[:, :, 2]
    #volumes = note_seq[:, :, 2]

    # The current pattern being played
    zeros_128 = np.zeros_like(replays[0])
    # Absolute tick of last event
    last_event_tick = 0
    # Amount of NOOP ticks
    noop_ticks = 0
    #record all turned-on notes
    notes_on=np.zeros_like(replays[0])
    for tick, data in enumerate(replays):
        replay = np.array(data)
        stop=stops[tick]
        if not np.array_equal(zeros_128, stop):
            noop_ticks = 0
            for index, is_stop in np.ndenumerate(stop):
                if is_stop==1:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=(tick - last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                    notes_on[index]=0

        if not np.array_equal(zeros_128, replay):
            noop_ticks = 0
            for index, next_volume in np.ndenumerate(replay):
                if next_volume>0:
                    # Was off, but now turned on or is being replayed
                    evt = midi.NoteOnEvent(
                        tick=(tick - last_event_tick) * step,
                        velocity=int(next_volume * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                    notes_on[index] = 1

        if np.array_equal(zeros_128, replay) and np.array_equal(zeros_128, stop):
            noop_ticks += 1

    tick += 1

    # Turn off all remaining on notes
    for index, is_on in np.ndenumerate(notes_on):
        if is_on > 0:
            # Was on, but now turned off
            evt = midi.NoteOffEvent(
                tick=(tick - last_event_tick) * step,
                pitch=index[0]
            )
            track.append(evt)
            last_event_tick = tick
            noop_ticks = 0

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=noop_ticks)
    track.append(eot)

    return pattern

def midi_decode(pattern,
                classes=MIDI_MAX_NOTES,
                step=None):
    """
    Takes a MIDI pattern and decodes it into a piano roll.
    """
    #load tone of Grand piano
    dataFile = os.path.join(CODE_DIR, 'tone_Grand_piano.mat')
    data = scio.loadmat(dataFile)
    tone = data['tone']

    #tick per step
    if step is None:
        step = pattern.resolution // NOTES_PER_BEAT

    # Extract all tracks at highest resolution
    merged_replay = None
    merged_volume = None
    merged_stop = None
    tempo_track=None
    cur_tempo_index = None
    target_tick = None
    for track in pattern:
        #ignore tracks that do not include any notes
        is_note_track=0
        is_tempo_track=0
        for i, event in enumerate(track):
            if isinstance(event, midi.NoteOnEvent) or isinstance(event, midi.NoteOffEvent):
                is_note_track = 1
                break
            if isinstance(event, midi.SetTempoEvent):
                is_tempo_track = 1
        if not is_note_track and not is_tempo_track:
            continue
        elif is_tempo_track and not is_note_track:
            tempo_track=track
            continue
        #time and tick passed
        time=0
        tick=0
        #two time step notations
        last_step = 0
        this_step = 0
        #ms per tick
        mptick=0
        if tempo_track is not None:
            cur_tempo_index = 0
            target_tick = 0
            for i, event in enumerate(tempo_track):
                if event.tick != 0:
                    cur_tempo_index=i
                    break
                if isinstance(event,midi.SetTempoEvent):
                    mpqn = event.mpqn
                    mptick = mpqn / pattern.resolution
            target_tick=tempo_track[cur_tempo_index].tick
        #record time after notes is played
        played_steps=np.zeros(classes)
        #record replay velocities
        velocities=np.zeros(classes)
        stop_notes=np.zeros(classes)
        # The downsampled sequences
        replay_sequence = []
        volume_sequence = []
        stop_sequence=[]

        for index, event in enumerate(track):
            #Count event ticks first...
                #for the case that the midi has a special track to record all tempo events
            if tempo_track is not None:
                for ticks in range(event.tick):
                    #check for the next tempo event when time is up
                    if tick==target_tick:
                        if isinstance(tempo_track[cur_tempo_index], midi.SetTempoEvent):
                            mpqn = tempo_track[cur_tempo_index].mpqn
                            mptick = mpqn / pattern.resolution
                        if not isinstance(tempo_track[cur_tempo_index], midi.EndOfTrackEvent):
                            cur_tempo_index += 1
                            while tempo_track[cur_tempo_index].tick == 0:
                                if isinstance(tempo_track[cur_tempo_index], midi.SetTempoEvent):
                                    mpqn = tempo_track[cur_tempo_index].mpqn
                                    mptick = mpqn / pattern.resolution
                                if isinstance(tempo_track[cur_tempo_index], midi.EndOfTrackEvent):
                                    target_tick = -1
                                    break
                                cur_tempo_index += 1
                            if target_tick > 0:
                                target_tick += tempo_track[cur_tempo_index].tick
                        else:target_tick=-1

                    #record notes according to time steps
                    this_step = round(time / 50000)
                    if last_step < this_step:
                        for i in range(this_step - last_step):
                            replay_now = get_replay(played_steps, velocities)
                            replay_sequence.append(replay_now)
                            stop_now = stop_notes
                            stop_sequence.append(stop_now)
                            volume_now = get_volume(velocities, played_steps, tone)
                            volume_sequence.append(volume_now)
                            #update played_steps, last_step, stop_notes
                            last_step = this_step
                            played_steps = update_played(played_steps)
                            stop_notes=np.zeros(classes)
                    #update time and tick
                    time += mptick
                    tick += 1

            #for the case that no special track to record all tempo events
            else:
                # Duplicate the last note pattern to wait for next event
                for ticks in range(event.tick):
                    time += mptick
                    this_step = round(time / 50000)
                    if last_step < this_step:
                        for i in range(this_step - last_step):
                            replay_now = get_replay(played_steps, velocities)
                            replay_sequence.append(replay_now)
                            stop_now = stop_notes
                            stop_sequence.append(stop_now)
                            volume_now = get_volume(velocities, played_steps, tone)
                            volume_sequence.append(volume_now)
                            # update played_steps, last_step, stop_notes
                            last_step = this_step
                            played_steps = update_played(played_steps)
                            stop_notes = np.zeros(classes)


            if isinstance(event, midi.EndOfTrackEvent):
                break
            # 3/11:take tempo info into consideration, quantize: 0.05s
            if isinstance(event, midi.SetTempoEvent):
                mpqn = event.mpqn
                mptick = mpqn / pattern.resolution
            # Modify the last note pattern
            if isinstance(event, midi.NoteOnEvent):
                pitch, velocity = event.data
                velocities[pitch]=velocity/MAX_VELOCITY
                played_steps[pitch]=1
                if velocity == 0:
                    played_steps[pitch]=-1
                    stop_notes[pitch]=1


            if isinstance(event, midi.NoteOffEvent):
                pitch, velocity = event.data
                velocities[pitch]=0
                played_steps[pitch]=-1
                stop_notes[pitch] = 1


        volume_sequence = np.array(volume_sequence)
        replay_sequence = np.array(replay_sequence)
        stop_sequence = np.array(stop_sequence)
        assert len(volume_sequence) == len(replay_sequence)
        assert len(volume_sequence) == len(stop_sequence)

        if merged_volume is None:
            merged_replay = replay_sequence
            merged_volume = volume_sequence
            merged_stop = stop_sequence
        else:
            # Merge into a single track, padding with zeros of needed
            if len(volume_sequence) > len(merged_volume):
                # Swap variables such that merged_notes is always at least
                # as large as play_sequence
                tmp = replay_sequence
                replay_sequence = merged_replay
                merged_replay = tmp

                tmp = volume_sequence
                volume_sequence = merged_volume
                merged_volume = tmp

                tmp = stop_sequence
                stop_sequence = merged_stop
                merged_stop = tmp

            assert len(merged_volume) >= len(volume_sequence)

            diff = len(merged_volume) - len(volume_sequence)
            merged_replay += np.pad(replay_sequence, ((0, diff), (0, 0)), 'constant')
            merged_volume += np.pad(volume_sequence, ((0, diff), (0, 0)), 'constant')
            merged_stop += np.pad(stop_sequence, ((0, diff), (0, 0)), 'constant')

    merged = np.stack([np.ceil(merged_replay),merged_stop, merged_replay, merged_volume], axis=2)
    # Prevent stacking duplicate notes to exceed one.
    merged = np.minimum(merged, 1)
    return merged

def load_midi(fname):

    p = midi.read_midifile(fname)
    cache_path = os.path.join(CACHE_DIR, fname[DATA_DIR_LENGTH:] + '.npy')
    print("loading " + fname + " to "+cache_path)
    sys.stdout.flush()
    try:
        note_seq = np.load(cache_path)
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        note_seq = midi_decode(p)
        np.save(cache_path, note_seq)

    assert len(note_seq.shape) == 3, note_seq.shape
    assert note_seq.shape[1] == MIDI_MAX_NOTES, note_seq.shape
    assert note_seq.shape[2] == 4, note_seq.shape
    assert (note_seq >= 0).all()
    assert (note_seq <= 1).all()
    return note_seq

if __name__ == '__main__':
    # Test
    # p = midi.read_midifile("out/test_in.mid")
    p = midi.read_midifile("out/test_in.mid")
    p = midi_encode(midi_decode(p))
    midi.write_midifile("out/test_out.mid", p)
