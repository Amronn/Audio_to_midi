import librosa
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav', 'piano_test.wav', 'test_piano_a0_c2.wav', 'test_5.wav']
file_path = 'wav_sounds/'+file_name[5]
model_o = load_model('models/octaves_cqt.h5')
hop_length = 256
y, sr = librosa.load(file_path)

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=85*3, n_bins=85*3*7, hop_length=hop_length))
threshold = 0.00
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length, norm=1)


def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 20)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
onset_samples_cqt = (onset_samples/hop_length).astype(int)

chroma=[]
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])

octave = []
for chromas in chroma:
    chromas = np.mean(chromas, axis = 1)
    pr = model_o.predict(np.array([[chromas]]))
    octave.append(np.argmax(pr))
    
print(octave)