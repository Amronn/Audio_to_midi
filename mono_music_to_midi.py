import librosa, librosa.display
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os


model = load_model('project_audio_to_midi/Audio_to_midi/notes_88.h5')
file_path = 'project_audio_to_midi/Audio_to_midi/notes_88/test.wav'

x, sr = librosa.load(file_path)

def fourier_pitch(segment, sr=sr, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    note = np.rint(librosa.hz_to_midi(f[np.argmax(X)])).astype(int)
    return note

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
        folder_names.append(i+21)
    return folder_names

labels = get_folder_names('project_audio_to_midi/Audio_to_midi/notes_88')
# labels = get_folder_names2(88)
print(labels)

def get_pitch(segment):
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, hop_length=256, n_mels=128, n_fft=512)
    scaler = MinMaxScaler()
    mel_spec = scaler.fit_transform(mel_spec)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    prediction = model.predict(mel_spec)
    pitch_index = np.argmax(prediction)
    return pitch_index

oenv = librosa.onset.onset_strength(y=x, sr=sr, aggregate=np.mean, detrend=True)
oenv[oenv < (np.max(oenv) / 10)] = 0
onset_samples = librosa.onset.onset_detect(y=x, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
onset_samples = np.concatenate([onset_samples, np.array([len(x)-1])])
segment_size = int(sr * 0.1) # długość jako 100ms
segments = np.array([x[i:i + segment_size] for i in onset_samples], dtype=object)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
librosa.display.waveshow(x,sr=sr)
plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1)
plt.show()


# Initialize an empty list to store the pitches for each segment
pitches_list = []

# Process each segment and collect the pitch in the list
for segment in segments:
    pitch = get_pitch(segment)
    #pitch = fourier_pitch(segment)
    pitches_list.append(int(labels[pitch]))


print(pitches_list)

timesx = librosa.samples_to_time(onset_samples)
timesx
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))
print(times)




from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

mid.save('audio_to_midi_test.mid')


