import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.signal
file_path = 'wav_sounds/liszt_frag.wav'
file_path = 'wav_sounds/piano_sample_2.wav'
file_path = 'wav_sounds/piano_chords_scale_in_C.wav'
y, sr = librosa.load(file_path)


def fourier_pitch(segment, sr=sr, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment, int(fmax - fmin)))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X)) # Use linspace with len(X) instead of fmax - fmin
    note = np.rint(librosa.hz_to_midi(f[np.argmax(X)])).astype(int)
    f0 = 32.70
    a = 2**(1/12)
    notes = [f0 * a**n for n in range(85)]
    g = np.searchsorted(f, notes)
    new_note = np.zeros_like(f)
    new_note[g] = 1
    new_note = np.correlate(new_note, [0.1, 0.8, 1, 0.8, 0.1], 'full')[4:]
    print(new_note)
    new_note[new_note>1] = 1
    X*=new_note
    plt.plot(f, X)
    plt.show()
    #przez przypadek prawie wynalazłem od nowa chromagram...
    return note

def fourier_pitch2(segment, sr=sr, num_of_harmonics = 10, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    avr = 0
    while avr<np.max(X)/5:
        avr = np.mean(X[X>0])
        X[X<avr] = 0
    first = 0
    for i in range(len(X)-1):
        if X[i+1]>X[i]:
            first = i+1
            break
    print(f[first])
    avr = np.ones_like(f)*avr
    plt.plot(f, X)
    plt.plot(f, avr)
    plt.show()
    note = np.rint(librosa.hz_to_midi(f[first])).astype(int)
    return note
oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
oenv[oenv < (np.max(oenv) / 100)] = 0
onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=False, units='samples').astype(int)
onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
segment_size = int(sr * 0.05)
segments = np.array([y[i:i + segment_size] for i in onset_samples], dtype=object)

all_hits = np.concatenate([onset_samples, np.array([len(y) - 1])])

# import matplotlib.pyplot as plt

# plt.figure(figsize=(15,6))
# librosa.display.waveshow(y,sr=sr)
# plt.vlines(librosa.samples_to_time(all_hits), ymin=-1, ymax=1)
# plt.show()

pitches_list = []

for segment in segments:
    pitch = fourier_pitch(segment)
    pitches_list.append(pitch)

# print(pitches_list)

timesx = librosa.samples_to_time(all_hits)
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

# print(times)


from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0)) # velocity ustawione jak narazie na stałe
    # print(time)
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time)) # tutaj nie ma znaczenia

mid.save('fourier_audio_to_midi.mid')