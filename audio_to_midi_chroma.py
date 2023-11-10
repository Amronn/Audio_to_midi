import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal

file_path = 'Audio_to_midi/wav_sounds/xd1.wav'
# file_path = 'wav_sounds/piano_2_notes.wav'
hop_length = 128
y, sr = librosa.load(file_path)

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.3
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)

librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
plt.show()

oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, detrend=True)
oenv[oenv < (np.max(oenv) / 10)] = 0
onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
segment_size = int(sr * 0.1) # długość jako 100ms
segments = np.array([y[i:i + segment_size] for i in onset_samples], dtype=object)

# print(onset_samples.shape)

onset_samples_cqt = (onset_samples/hop_length).astype(int)
# print(onset_samples_cqt.shape)
chroma=[]
for i in onset_samples_cqt:
    chroma.append(chroma_orig[:, i:i+10])

index = 0

data = chroma[index]

# librosa.display.specshow(data, y_axis='chroma', x_axis='time', sr=sr)
# plt.show()
chroma_av = []
for chroma in chroma:
    # print(chroma.shape)
    chroma_av.append(np.mean(chroma, axis=1))
    print(librosa.midi_to_note(np.argmax(np.mean(chroma, axis=1))+24))


# print(librosa.midi_to_note(np.argmax(chroma_av[index])+24))

# t = np.linspace(24, 108, 85)

# plt.plot(chroma_av[index], t)
# plt.grid()
# plt.show()

pitches_list = []
for chroma_av in chroma_av:
    pitches_list.append(np.argmax(chroma_av)+24)


# print(pitches_list)

timesx = librosa.samples_to_time(onset_samples)
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))
# print(times)

from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

mid.save('cqt_chroma_midi.mid')

