import librosa
import numpy as np

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

def chroma(hop_length, y, sr, keys, fmin = 32.7, n_octaves = 7):
    C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
    chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=keys, bins_per_octave=keys*3, fmin = fmin, n_octaves=n_octaves, hop_length=hop_length, norm=1)
    return chroma_orig