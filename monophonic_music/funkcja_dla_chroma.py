import numpy as np, librosa
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

def correct_pitch(chroma, threshold=0.1, wiener_size=3):
    chroma = scipy.signal.wiener(chroma, wiener_size)
    threshold = 0.1
    chroma[chroma<threshold]=0
    peaks, what = scipy.signal.find_peaks(chroma)
    plt.plot(chroma)
    plt.vlines(peaks, ymin=0, ymax=1)
    plt.show()
    #korekta wysokości w zależności od wpływu harmonicznych. Zakładam, że dźwiękiem podstawowym jest pierwsza harmoniczna. 
    pitch = 0
    if len(peaks)>1:
        if (peaks[1]-peaks[0])%12 != 0: #druga harmoniczna mocniejsza
            pitch = peaks[1]-12
        elif (peaks[1]-peaks[0])%12 == 0: #druga harmoniczna mocniejsza
            pitch = peaks[1]-12
    else:
        pitch = np.argmax(chroma)
    return pitch

def alikwot_check(chroma):
    alikwoty = []
    for i in range(85):
        a = np.zeros(85)
        indices = [i, i + 12, i + 12 + 7, i + 12 + 7 + 5]
        indices = [idx for idx in indices if idx < 85]
        a[indices] = 1
        alikwoty.append(a)
    return alikwoty
 
file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav']
file_path = 'Audio_to_midi/wav_sounds/'+file_name[1]
file_path = 'Audio_to_midi/wav_sounds/piano_test2.wav'
hop_length = 256
y, sr = librosa.load(file_path)

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.0
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)

onset_samples = get_onsets(y, sr)
cqt_points = (onset_samples/hop_length).astype(int)

chroma = chroma_orig[:,cqt_points[8]:cqt_points[9]-10]

# librosa.display.specshow(data = chroma, y_axis='chroma', x_axis='time', sr=sr,hop_length=hop_length)
# plt.show()

chroma = np.mean(chroma, axis = 1)



print(correct_pitch(chroma)+24)
# t = range(85)

