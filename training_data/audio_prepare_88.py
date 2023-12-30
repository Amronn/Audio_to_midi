import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
number_of_keys = 85
number_of_samples = 4

note_names = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs','g','gs', 'a','as','b']

file_path = 'E:\Amron\music_processing\project_audio_to_midi\piano_notes_neo_piano.wav'
# file_path = 'wav_sounds\piano_a0_c2.wav'
hop_length = 512
y, sr = librosa.load(file_path)
# y = librosa.effects.harmonic(y)
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.0
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length, norm=1)
# librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
# plt.show()
def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True, center = False)
    # plt.plot(oenv)
    # plt.show()
    oenv[oenv < (np.max(oenv) / 20)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
# to_delete = []
# for i in range(len(onset_samples)-1):
#     if onset_samples[i+1] - onset_samples[i] < 7000:
#         to_delete.append(i)
        
# onset_samples = np.delete(onset_samples, to_delete)
# to_delete = []
# for i in range(len(onset_samples)-1):
#     if onset_samples[i+1] - onset_samples[i] < 17000:
#         to_delete.append(i+1)
# onset_samples = np.delete(onset_samples, to_delete)

# for i in range(len(onset_samples)-1):
#     if onset_samples[i+1] - onset_samples[i] > 36000:
#         onset_samples = np.append(onset_samples,(onset_samples[i]+onset_samples[i+1])//2)
# onset_samples = np.append(onset_samples, [9*sr,53*sr, int(150.8*sr), 168*sr, 180*sr, int(200.7*sr), 216*sr, 217*sr, int(221.5*sr)])
onset_samples = np.unique(onset_samples)
onset_samples_cqt = onset_samples//hop_length
print(len(onset_samples))
# onset_samples = [0]
# onset_samples_cqt = [0]

# for i in range(number_of_keys*number_of_samples):
#     onset_samples.append(int(i*sr))
#     onset_samples_cqt.append(int(i*sr/hop_length))

# print(onset_samples_cqt)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
librosa.display.waveshow(y,sr=sr)
plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1, color='red')
plt.show()
plt.figure(figsize=(15, 6))
librosa.display.waveshow(y, sr=sr)

# Dodanie czerwonych pionowych linii
plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1, color='red')

# Dodanie oznaczeń na osi x w jednostkach czasu
time_labels = [f"{librosa.samples_to_time(sample):.2f}s" for sample in onset_samples]
plt.xticks(librosa.samples_to_time(onset_samples), time_labels, rotation=45)

plt.show()
chroma = []
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]-10])

chroma_av = []
for chroma in chroma:
    chroma_av.append(np.mean(chroma, axis=1))
    
# print(len(chroma_av))

for i in range(number_of_keys):
    folder_path = f'notes_88_v3_cqt/{i}'
    os.makedirs(folder_path, exist_ok=True)
    for k in range(number_of_samples):
        f_name = f'chroma_{note_names[i%12]}{i//12}{k}.csv'
        file_path = os.path.join(folder_path, f_name)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            data = chroma_av[i*4+k]
            writer.writerows(map(lambda x: [x],data))

