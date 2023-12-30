import numpy as np
import scipy.signal
import scipy.ndimage
import librosa
import matplotlib.pyplot as plt
import keras
# file_path = 'wav_sounds/piano_sample_2.wav'
file_path = 'wav_sounds/liszt_frag.wav'
# file_path = 'wav_sounds/piano_chords_scale_in_C.wav'
x, sr = librosa.load(file_path)

num = 1
    
def fourier_pitch(segment, sr=sr, num_of_harmonics = 10, fmin=16, fmax=8192):
    N = len(segment)
    print(N)
    X = np.abs(np.fft.fft(segment))
    print(len(X))
    f = np.linspace(0, sr, len(X))