import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

file_path = 'wav_sounds/chords.wav'



y, sr = librosa.load(file_path)


oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
oenv[oenv < (np.max(oenv) / 10)] = 0
onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples')
segment_size = int(sr * 0.1) # długość jako 100ms
segments = np.array([y[i:i + segment_size] for i in onset_samples])

hop_length = 256

onset_samples_cqt = (onset_samples/hop_length).astype(int)
print(onset_samples_cqt)

print(segments[0].shape)

C = np.abs(librosa.cqt(y=y, sr=sr, n_bins = 88, hop_length = hop_length))
print(C.shape)

C = C.T
C1 = np.array([C[i:i + int(segment_size/hop_length)] for i in onset_samples_cqt], dtype=object)

C1 = np.array([C1[i].T for i in range(len(C1))])
C = C.T
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(C.astype(float), ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()

print(C1[2][:10].shape)
scaler = MinMaxScaler()
cqt = scaler.fit_transform(C1[2][:10])
f_name = f'cqt_data_check.csv'
file_path = os.path.join('training_data', f_name)
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(cqt)
    
