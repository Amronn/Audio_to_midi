import cv2
import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from mido import Message, MidiFile, MidiTrack
file_path = 'wav_sounds/piano_2_notes.wav'
file_path = 'wav_sounds\piano_test2.wav'
# file_path = 'wav_sounds\piano_chords_123.wav'
# file_path = 'wav_sounds\liszt_frag.wav'
hop_length = 256
y, sr = librosa.load(file_path)

num = 0

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.0
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)
chroma_orig = scipy.signal.convolve2d(chroma_orig, np.ones((1,3)))
chroma_orig = scipy.signal.convolve2d(chroma_orig, np.ones((1,10))*2)
chroma_orig /= np.max(chroma_orig)
# chroma_orig[chroma_orig<0.3] = 0

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

def alikwot_check(chroma):
    chroma /= np.max(chroma)
    har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
    harmoniczne = np.zeros(85)
    # k=0
    # for i in har[:num_of_harmonics]:
    #     # harmoniczne[i] = 1/(2*k+1)
    #     harmoniczne[i] = 1
        # k=k+1
        
    cor = np.zeros(85)
    for n in range(1,len(har)):
        harmoniczne = np.zeros_like(cor)
        for i in har[:n]:
            harmoniczne[i] = 1
        cor += np.correlate(chroma, harmoniczne, 'full')[84:]
    
    # if np.argmax(cor)*2>=85:
    return np.argmax(cor)+24
    # else:
    #     for i in range(np.argmax(cor)*2):
    #         cor[i] = 0
    plt.plot(cor)
    plt.show()

    
    # check = np.correlate(chroma, harmoniczne, 'full')[84:]
    # chroma[chroma<0.1*np.max(chroma)] = 0
    # checked = [chroma[i]*check[i] for i in range(len(chroma))]
    # number = []
    # for i in range(len(chroma)):
    #     if chroma[i]!=0:
    #         if chroma[i]*check[i]>0:
    #             number.append(i)
    
    
    
    # plt.subplot(3,1,1),plt.plot(chroma, label = 'chroma_av'), plt.legend()
    # plt.subplot(3,1,2), plt.plot(harmoniczne, label = 'harmoniczne'), plt.legend()
    # plt.subplot(3,1,3), plt.plot(checked, label ='check'), plt.legend()
    # plt.subplot(3,1,3), plt.vlines(number, ymin=-1, ymax=1, colors='red')
    # plt.show()
    # checked = np.array(checked)
    # checked[checked<1.0] = 0
    # peaks, _ = scipy.signal.find_peaks(checked)
    cor[cor>1] = 1
    peaks, _ = scipy.signal.find_peaks(cor)
    midi_notes = [peak + 24 for peak in peaks]
    
    return midi_notes

onset_samples = get_onsets(y, sr)
onset_samples = np.unique(onset_samples)

onset_samples_cqt = (onset_samples/hop_length).astype(int)
print(onset_samples_cqt)

# for num in range(0,len(onset_samples)-1):

chroma = chroma_orig[:, onset_samples_cqt[num]:onset_samples_cqt[num+1]]
chroms = np.mean(chroma, axis=1)
print(alikwot_check(chroms))

# librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
# plt.grid()
# plt.show()

# # Wczytaj macierz chromagramu (przykład - możesz dostosować do swoich danych)
chromagram_matrix = chroma

chromagram_matrix = scipy.signal.convolve2d(chroma, np.ones((1,100))*10)

chromagram_matrix /= np.max(chromagram_matrix)


# Zastosuj algorytm Canny do macierzy chromagramu
edges = cv2.Canny((chromagram_matrix * 255).astype(np.uint8), 100, 255)

# Wyświetl obraz oryginalny i wynikowy
plt.subplot(121), librosa.display.specshow(chromagram_matrix, y_axis='chroma', x_axis='time')
plt.title('Chromagram'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Krawędzie po Canny'), plt.xticks([]), plt.yticks([])

plt.show()