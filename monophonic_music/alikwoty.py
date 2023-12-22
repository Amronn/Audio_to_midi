import numpy as np, librosa
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
import csv
csv_path = 'alikwoty.csv'
def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    # oenv = scipy.signal.wiener(oenv, 10)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples


num = 2

def alikwot_check(chroma, num_of_harmonics = 3):

    peaks, what = scipy.signal.find_peaks(chroma)
    x = range(85)
    plt.plot(chroma)
    plt.show()
    print(chroma[12])
    if len(peaks)==1:
        return int(peaks[0])+24

    if len(peaks)>1:
        har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
        harmoniczne = np.zeros(85)
        harmoniczne[har[:4]] = 1
        harmonics_group = []
        tr = 0.1
        for i in range(85):
            if i+24<84:
                als = np.array([chroma[i], chroma[i+12], chroma[i+19], chroma[i+24]])
                harmonics_group.append(als)
                # if chroma[i]>tr:
                    
            elif i+19<85:
                als = np.array([chroma[i], chroma[i+12], chroma[i+19], 0])
                harmonics_group.append(als)
            elif i+12<85:
                als = np.array([chroma[i], chroma[i+12], 0, 0])
                harmonics_group.append(als)
            else:
                harmonics_group.append(np.array([chroma[i], 0, 0, 0]))


        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            data = harmonics_group
            writer.writerows(map(lambda x: [x],data))
        
        librosa.display.specshow(data = np.array(harmonics_group), y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length)
        plt.show()
        # x = range(0,85)
        # plt.plot(x, harmoniczne, 'o', label='Dane punkty')
        # plt.legend()
        # plt.show()
        # check = np.correlate(chroma, harmoniczne, 'full')
        # plt.plot(check[84:])
        # plt.show()
        # return np.argmax(check[84:])+24


file_path = 'Audio_to_midi/wav_sounds/liszt_frag.wav'
# file_path = 'Audio_to_midi/wav_sounds/piano_test2.wav'
file_path = 'Audio_to_midi/wav_sounds/piano_test.wav'
file_path = 'wav_sounds/piano_chords_123.wav'
file_path = 'wav_sounds/casio_c.wav'
file_path = 'wav_sounds/casio_chords.wav'

hop_length = 256
y, sr = librosa.load(file_path)

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.0
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)

onset_samples = get_onsets(y, sr)
onset_samples = np.unique(onset_samples)
cqt_points = (onset_samples/hop_length).astype(int)
print(cqt_points)
plt.figure(figsize=(15,6))
librosa.display.waveshow(y,sr=sr)
plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1)
plt.show()

chroma = chroma_orig[:,cqt_points[num-1]:cqt_points[num]]

# librosa.display.specshow(data = chroma, y_axis='chroma', x_axis='time', sr=sr,hop_length=hop_length)
# plt.show()

# chroma = np.mean(chroma, axis = 1)
def avr_w(data):
    n = len(data)
    weights = [np.exp(-i/10)*10 for i in range(1, n+1)]
    average_with_weights = sum(w * x for w, x in zip(weights, data)) / sum(weights)
    return average_with_weights
chromas = []
for ch in chroma:
    chromas.append(avr_w(ch))

pitch = alikwot_check(chromas)
print(pitch)

