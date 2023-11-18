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


def alikwot_check(num_of_harmonics = 3):
    har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]

    alikwoty = []
    for i in range(85):
        a = np.zeros(85)
        indices = [i, i + 12, i + 12 + 7, i + 12 + 7 + 5]
        indices = [i + har[k] for k in range(num_of_harmonics+1)]
        indices = [idx for idx in indices if idx < 85]
        a[indices] = 1
        alikwoty.append(a)
    check = []
    for i in range(85):
        cor = np.correlate(alikwoty[i], chroma)
        check.append(cor)
        
    return np.argmax(check)+24

def generate_harmonic_indices(num_harmonicznych):
    harmonic_indices = [i for i in range(0, num_harmonicznych * 24, 24)]
    return harmonic_indices

print(generate_harmonic_indices(7))

file_path = 'Audio_to_midi/wav_sounds/liszt_frag.wav'
hop_length = 256
y, sr = librosa.load(file_path)

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.0
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)

onset_samples = get_onsets(y, sr)
cqt_points = (onset_samples/hop_length).astype(int)

chroma = chroma_orig[:,cqt_points[0]:cqt_points[1]-10]

# librosa.display.specshow(data = chroma, y_axis='chroma', x_axis='time', sr=sr,hop_length=hop_length)
# plt.show()

chroma = np.mean(chroma, axis = 1)

pitch = alikwot_check()
print(pitch)

har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
x = range(len(har))

plt.plot(x, har)
plt.show()

