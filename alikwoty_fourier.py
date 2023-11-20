import numpy as np
import librosa
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import csv
from scipy.fft import fft, fftfreq
file_path = 'Audio_to_midi/wav_sounds/piano_test2.wav'
csv_path = 'Audio_to_midi/fourier_test.csv'
y, sr = librosa.load(file_path)

def fourier_pitch(segment, sr=sr, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    note = np.rint(librosa.hz_to_midi(f[np.argmax(X)])).astype(int)
    return note

def fourier_pitch2(segment, sr=sr, num_of_harmonics = 10):

    N = len(segment)
    T = 1.0 / sr
    X = fft(segment, norm='forward')
    X = np.abs(X[0:N//2])
    # m2 = max(X)
    # X[X<=m2/10] = 0
    # X[X>m2/10] = 1
    # print(len(X))
    # plt.plot(X)
    # plt.show()
    checks = []
    f0 = 32.70
    a = 2**(1/12)
    notes = [f0*a**n for n in range(85)]
    print(notes)
    for note in notes:
        harmoniczne = [k*note for k in range(1,num_of_harmonics+1) if k*note<len(X)]
        x = np.linspace(0,1,N)
        signal = 0
        if len(harmoniczne)>0:
            for h in harmoniczne:
                signal = signal+np.sin(2*np.pi*h*x)
            X2 = fft(signal, norm='forward')
            X2 = np.abs(X2[0:N//2])
            # m = max(X2)
            # X2[X2<=m/10] = 0
            # X2[X2>m/10] = 1
            # print(len(X2))
            # plt.plot(X2)
            # plt.show()
            X2 = np.convolve(X2, np.ones(5), mode='same')
            check = np.correlate(X, X2, 'valid')
            checks.append(check)
    with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            data = checks
            writer.writerows(map(lambda x: [x],data))
    print(np.argmax(checks))
    
def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)

# number_of_segment = int(input())
number_of_segment = 1

segment = y[onset_samples[number_of_segment-1]:onset_samples[number_of_segment]]

pitch = fourier_pitch2(segment)

print(pitch)
