import librosa
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os


model = load_model('Audio_to_midi/notes_88_cqt.h5')

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

# labels = get_folder_names('Audio_to_midi/notes_88_cqt')
labels = get_folder_names2(88)
# print(labels)

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav', 'piano_test.wav']
file_path = 'Audio_to_midi/wav_sounds/'+file_name[2]
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
print(onset_samples)

onset_samples_cqt = (onset_samples/hop_length).astype(int)
# print(onset_samples_cqt)
chroma=[]
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])



def get_pitch(segment):
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, hop_length=256, n_mels=128, n_fft=512)
    scaler = MinMaxScaler()
    mel_spec = scaler.fit_transform(mel_spec)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    prediction = model.predict(mel_spec)
    pitch_index = np.argmax(prediction)
    return pitch_index

def get_pitch2(chromas):
    print(chromas.shape)
    prediction = model.predict(np.array([chromas]))
    pitch_index = np.argmax(prediction)+24
    return pitch_index

pitches_list = []
chroma_av = []
for chroma in chroma:
    chroma_av.append(np.mean(chroma, axis=1))
    print(np.sum(chroma, axis=1).shape)
    pitches_list.append(get_pitch2(np.expand_dims(np.mean(chroma, axis=1), axis=0)))

timesx = librosa.samples_to_time(onset_samples)

times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

# import matplotlib.pyplot as plt

# plt.plot(timesx, pitches_list)
# plt.show()

from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

mid.save('audio_to_midi_test.mid')



