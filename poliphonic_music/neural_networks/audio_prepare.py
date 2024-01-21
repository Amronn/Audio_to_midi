import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import fun
file_path = 'wav_sounds/chords_neo_piano_1000.wav'
hop_length = 512
y, sr = librosa.load(file_path)

chroma_orig = fun.chroma(hop_length, y, sr, 88)
onset_samples = fun.get_onsets(y, sr)
onset_samples_cqt = onset_samples//hop_length

chroma = []
# for i in range(len(onset_samples_cqt)-1):
#     chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]-10])
for i in range(len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i]+32])
# print(chroma.shape)
chroma_av = []
for chroma1 in chroma:
    me = np.mean(chroma1, axis=1)
    # me/=np.max(me)
    # me[me<0.1] = 0
    chroma_av.append(me)
    

print(len(onset_samples))
print(len(chroma_av))

# librosa.display.specshow(chroma_orig[:10000], y_axis='chroma', x_axis='time', sr=sr)
# plt.show()
# plt.figure(figsize=(15,6))
# librosa.display.waveshow(y,sr=sr)
# plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1, color='red')
# plt.show()
# plt.plot(chroma_av[0])
# plt.grid()
# plt.show()

folder_path = f'poliphonic_music/chords_cqt_1000'
os.makedirs(folder_path, exist_ok=True)
for i in range(1000):
    f_name = f'chroma_{i}.csv'
    file_path = os.path.join(folder_path, f_name)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        data = chroma[i]
        writer.writerows(data)

