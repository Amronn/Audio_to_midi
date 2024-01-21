import librosa
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os

import matplotlib.pyplot as plt

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav', 'piano_test.wav', 'piano_test.wav','test_piano_a0_c2.wav', 'test_5.wav']
file_path = 'wav_sounds/'+file_name[0]

notes = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g','gs','a','as','h']

hop_length = 256
y, sr = librosa.load(file_path)

model_o = load_model('models/octaves_cqt.h5')
model_n = load_model('models/notes_12_cqt.h5')

pitches_list = []
octave = []
note = []
chroma=[]
chroma2=[]

def get_pitch_chromas(chroma, chroma2, pitches_list):
    for ch in chroma:
        octave.append(np.argmax(model_o.predict(np.array([[np.mean(ch, axis=1)]]))))
    for ch2 in chroma2:
        note.append(np.argmax(model_n.predict(np.array([[np.mean(ch2, axis=1)]]))))
    for i in range(len(octave)):
        pitches_list.append(octave[i]*12+note[i]+24)

threshold = 0.0
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length, norm=1)
chroma_orig2 = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=12, bins_per_octave=12*3, threshold=threshold, hop_length=hop_length, norm=1)

# import matplotlib.pyplot as plt
librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
plt.show()

librosa.display.specshow(chroma_orig2, y_axis='chroma', x_axis='time', sr=sr)
plt.show()

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)

onset_samples_cqt = (onset_samples/hop_length).astype(int)


for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])

for i in range(len(onset_samples_cqt)-1):
    chroma2.append(chroma_orig2[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])
    
get_pitch_chromas(chroma, chroma2, pitches_list)

timesx = librosa.samples_to_time(onset_samples)

times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

mid.save('monophonic_music/audio_to_midi/cqt_neural_to_midi.mid')




