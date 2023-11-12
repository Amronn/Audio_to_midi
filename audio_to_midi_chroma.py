import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from mido import Message, MidiFile, MidiTrack

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav']
file_path = 'Audio_to_midi/wav_sounds/'+file_name[1]
hop_length = 256
y, sr = librosa.load(file_path)
y = librosa.effects.harmonic(y)
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length, filter_scale=0.6, sparsity=0.05))
threshold = 0.5
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)



def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.average, detrend=True)
    # oenv[oenv < (np.max(oenv) / 100)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)

librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
plt.show()

# plt.figure(figsize=(15,6))
# librosa.display.waveshow(y,sr=sr)
# plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1)
# plt.show()


onset_samples_cqt = (onset_samples/hop_length).astype(int)
# print(onset_samples_cqt)
chroma=[]
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])

chroma_av = []
for chroma in chroma:
    chroma_av.append(np.sum(chroma, axis=1))
    # print(librosa.midi_to_note(np.argmax(np.mean(chroma, axis=1))+24))


pitches_list = []
for chroma_av in chroma_av:
    pitches_list.append(np.argmax(chroma_av)+24)

timesx = librosa.samples_to_time(onset_samples)
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

def create_midi(pitches_list, times):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for i, time in enumerate(times):
        track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
        track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

    mid.save('Audio_to_midi/cqt_chroma_midi.mid')

create_midi(pitches_list, times)

