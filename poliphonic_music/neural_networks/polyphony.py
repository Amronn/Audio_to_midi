import librosa
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import fun

model = load_model('models/num_cqt.h5')

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav', 'piano_test.wav', 'piano_test.wav','test_piano_a0_c2.wav', 'test_5.wav']
file_path = 'wav_sounds/'+file_name[0]
# file_path = 'wav_sounds/chords.wav'
hop_length = 512
y, sr = librosa.load(file_path)

# chroma_orig = fun.chroma(hop_length, y, sr, 4*12+1, fmin = 65.41, n_octaves=4)
# onset_samples = fun.get_onsets(y, sr)
# onset_samples_cqt = onset_samples//hop_length
chroma_orig = fun.chroma(hop_length, y, sr, 88)
onset_samples = fun.get_onsets(y, sr)
onset_samples_cqt = onset_samples//hop_length
chroma = []
for i in range(len(onset_samples_cqt)-1):
    chroma.append(np.array(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i]+32]))
    
chroma_av = []
for chroma1 in chroma:
    me = np.mean(chroma1, axis=1)
    # me/=np.max(me)
    # me[me<0.1] = 0
    chroma_av.append(me)
    
print(chroma[0].shape)
num = 1
plt.plot(chroma_av[0])
# librosa.display.specshow(chroma[0].T, y_axis='chroma', x_axis='time', sr=sr)
# plt.grid()
plt.show()


#wyświetl wykresy predykcji i orignalnego chromagramu
i=0
notes = []
for ch in chroma:
    prediction = model.predict(np.array([ch]))
    for k in range(88):
        if prediction[0][k]>0.05:
            notes.append(k+21)
    # plt.plot(prediction[0])
    # plt.show()
    print(librosa.midi_to_note(notes))
    notes = []
    i+=1