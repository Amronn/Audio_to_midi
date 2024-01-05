import librosa
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from fun import get_onsets, chroma

model = load_model('models/num_cqt.h5')

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav', 'piano_test.wav', 'piano_test.wav','test_piano_a0_c2.wav', 'test_5.wav']
file_path = 'wav_sounds/'+file_name[0]

hop_length = 512
y, sr = librosa.load(file_path)

chroma_orig = chroma(hop_length, y, sr, 88)

onset_samples = get_onsets(y, sr)
onset_samples_cqt = onset_samples//hop_length

chrom = []
# for i in range(len(onset_samples_cqt)-1):
#     chrom.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]-10])
for i in range(len(onset_samples_cqt)-1):
    chrom.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i]+32].T)
    
chroma_av = []
for chroma1 in chrom:
    chroma_av.append(np.mean(chroma1, axis=1))

print(chrom[0].shape)
num = 1
prediction = model.predict(np.array([chrom[num]]))

plt.plot(prediction[0])
plt.plot(chrom[num].T, 'r')
plt.show()