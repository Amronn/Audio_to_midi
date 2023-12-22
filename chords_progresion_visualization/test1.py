import numpy as np
import librosa
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import csv
file_path = 'wav_sounds/piano_chords_scale_in_C.wav'
y, sr = librosa.load(file_path)

def fourier(segment, sr = 22050, fmin=16, fmax=8192, plot = True):
    N = len(segment)
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    if plot:
        plt.figure(figsize=(15,6))
        plt.plot(f[int(fmin/sr*N):int(fmax/sr*N)], X[int(fmin/sr*N):int(fmax/sr*N)])
        plt.xlabel('Frequency Hz')
        plt.show()
    return [X, f]
    
def chroma(y, sr = 22050, hop_length = 256, threshold = 0.0, convolve = False, convolution_size = 4, convolution_divide = 10, plot = True):
    C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
    chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)
    if convolve:
        chroma_orig = scipy.signal.convolve2d(chroma_orig, np.ones((1,convolution_size))/convolution_divide)
    if plot:
        librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
        plt.grid()
        plt.show()
    return chroma_orig

    
def get_onsets(y, sr=22050, plot=True):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    if plot:
        plt.figure(figsize=(15,6))
        librosa.display.waveshow(y,sr=sr)
        plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1)
        plt.show()
    return onset_samples

onset_samples = get_onsets(y, plot = False)

num = 1

segment = y[onset_samples[num-1]:onset_samples[num]]
X,f = fourier(segment)

chromagram = chroma(segment, plot = True)

chromagram = np.mean(chromagram, axis = 1)
def zero_one(data, threshold):
    data[data<threshold] = 0
    data[data>threshold] = 1
    return data

def only_good_ones(data, threshold, midi_offset = 24):
    new_data = []
    for i, data in enumerate(data):
        if data>threshold:
            new_data.append(i+midi_offset)
    return new_data

good_ones = only_good_ones(chromagram, np.max(chromagram)/2)
print(good_ones)

