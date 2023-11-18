import librosa
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os


model = load_model('Audio_to_midi/notes_88_cqt.h5')
model_m = load_model('Audio_to_midi/notes_88_mels.h5')

def get_folder_names(directory_path):
    folder_names = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folder_names.append(int(item))
    return folder_names

def get_folder_names2(num):
    folder_names = []
    for i in range(num):
        folder_names.append(i+24)
    return folder_names

def get_pitch_cqt(chromas):
    prediction = model.predict(np.array([chromas]))
    pitch_index = np.argmax(prediction)+24
    return pitch_index

def get_pitch_mel(mels):
    prediction = model_m.predict(np.array([mels]))
    pitch_index = np.argmax(prediction)+24
    return pitch_index

# labels = get_folder_names('Audio_to_midi/notes_88_cqt')
labels = get_folder_names2(88)
# print(labels)

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav', 'piano_test.wav']
file_path = 'Audio_to_midi/wav_sounds/'+file_name[0]
hop_length = 256
y, sr = librosa.load(file_path)
# y=librosa.effects.harmonic(y=y)
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length, filter_scale=0.6, sparsity=0.05))
threshold = 0.3
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 20)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
# print(onset_samples)

onset_samples_cqt = (onset_samples/hop_length).astype(int)
# print(onset_samples_cqt)
chroma=[]
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])

segment_size = int(sr * 0.05)
segments = np.array([y[i:i + segment_size] for i in onset_samples[:len(onset_samples)-1]])
mels = []
for seg in segments:
    mel_spect = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=2048, hop_length=256)
    mel_spect = np.mean(mel_spect, axis = 1)
    mels.append(mel_spect)

pitches_list = []

def get_pitch_chromas(get_pitch, chroma, pitches_list):
    for chroma in chroma:
        pitches_list.append(get_pitch(np.expand_dims(np.mean(chroma, axis=1), axis=0)))

def get_pitch_mels(get_pitch, mels, pitches_list):
    for mel in mels:
        pitches_list.append(get_pitch(np.expand_dims(mel, axis=0)))

# get_pitch_mels(get_pitch_mel, mels, pitches_list)

get_pitch_chromas(get_pitch_cqt, chroma, pitches_list)

timesx = librosa.samples_to_time(onset_samples)

times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

# import matplotlib.pyplot as plt

# plt.plot(timesx, pitches_list)
# plt.show()

from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

print(times)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

mid.save('audio_to_midi_test.mid')



