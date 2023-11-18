import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
number_of_keys = 85
number_of_samples = 4

note_names = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs','g','gs','a','as','b']

file_path = 'E:\Amron\music_processing\project_audio_to_midi\piano_notes_neo_piano.wav'
hop_length = 512
y, sr = librosa.load(file_path)
# y = librosa.effects.harmonic(y)
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length, filter_scale=0.6))
threshold = 0.1
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length, norm=1)
librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
plt.show()
def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 20)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)

onset_samples = [0]
onset_samples_cqt = [0]

for i in range(341):
    onset_samples.append(int(i*sr))
    onset_samples_cqt.append(int(i*sr/hop_length))

print(onset_samples_cqt)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
librosa.display.waveshow(y,sr=sr)
plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1, color='red')
plt.show()

chroma = []
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]-10])

chroma_av = []
for chroma in chroma:
    chroma_av.append(np.mean(chroma, axis=1))
    
# print(len(chroma_av))

for i in range(number_of_keys):    
    folder_path = f'Audio_to_midi/notes_88_cqt/{i}'
    os.makedirs(folder_path, exist_ok=True)
    for k in range(number_of_samples):
        f_name = f'chroma_{note_names[i%12]}{i//12}{k}.csv'
        file_path = os.path.join(folder_path, f_name)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            data = chroma_av[i*4+k]
            writer.writerows(map(lambda x: [x],data))

