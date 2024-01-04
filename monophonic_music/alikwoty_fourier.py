import numpy as np
import librosa
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import csv
# file_path = 'wav_sounds/piano_chords_123.wav'
# file_path = 'wav_sounds/casio_c.wav'
# file_path = 'wav_sounds/casio_chords.wav'
file_path = 'wav_sounds/piano_sample.wav'
csv_path = 'fourier_test.csv'

y, sr = librosa.load(file_path)

def fourier_pitch(segment, sr=sr, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    note = np.rint(librosa.hz_to_midi(f[np.argmax(X)])).astype(int)
    return note

number_of_segment = 1

def fourier_pitch2(segment, sr=sr, num_of_harmonics = 4, fmin=16, fmax=8192):
    N = len(segment)
    # print(N)
    X = np.abs(np.fft.fft(segment))
    X /= np.max(X)
    X = X[int(fmin/sr*N):int(fmax/sr*N)]
    # X[X>0.1]=1 
    # X[X<=0.1]=0
    # print(len(X))
    f = np.linspace(0, sr, len(X))
    plt.plot(f[int(fmin/sr*N):int(fmax/sr*N)], X[int(fmin/sr*N):int(fmax/sr*N)])
    plt.show()
    # X = scipy.signal.wiener(X, 3)
    peaks, what = scipy.signal.find_peaks(X)
    for i in range(40, len(X)):
        amp = X[i]
        k=i
        k+=k
        while k<len(X):
            X[k] -= amp
            k+=k
    
    plt.plot(f[int(fmin/sr*N):int(fmax/sr*N)], X[int(fmin/sr*N):int(fmax/sr*N)])
    plt.show()
    
    
    if len(peaks) == 1:
        return int(peaks[0])+24
    else:
        f0 = 32.70
        a = 2**(1/12)
        notes = [f0*a**n for n in range(85)]
        check = []
        # note = f0*a**(60-24)
        for note in notes:
            harmoniczne = [k*note for k in range(1,num_of_harmonics+1) if k*note<len(X)]
            # print(harmoniczne)
            x = np.linspace(0,N/sr,N)
            signal = 0
            if len(harmoniczne)>0:
                for h in harmoniczne:
                    signal = signal+np.cos(2*np.pi*h*x)
                X2 = np.abs(np.fft.fft(signal))
                X2 /= np.max(X2)
                X2[X2>0.5] = 1
                X2[X2<0.5] = 0
                f = np.linspace(0, sr, len(X2))
                # plt.plot(f[int(fmin/sr*N):int(fmax/sr*N)], X2[int(fmin/sr*N):int(fmax/sr*N)])
                # plt.show()
                
                cor = np.dot(X, X2.T)
                print(cor)

                check.append(cor)
        t = range(24,109)
        plt.plot(t, check)
        plt.show()
    return (np.argmax(check))+24
    # har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
    # harmoniczne = np.zeros(85)
    # k=0
    # for i in har[:7]:
    #     harmoniczne[i] = 1 + np.exp(-2*k)
    #     k=k+1
    # check = np.correlate(check, harmoniczne, 'full')[84:]
    # # plt.plot(check)
    # # plt.show()
    # check[check<np.max(check)/np.pi] = 0
    plt.figure(figsize=(15,6))
    plt.plot(f[int(fmin/sr*N):int(fmax/sr*N)], X[int(fmin/sr*N):int(fmax/sr*N)])
    plt.xlabel('Frequency Hz')
    plt.show()
    plt.plot(check)
    plt.show()
    with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            data = check
            writer.writerows(map(lambda x: [x],data))
    return np.argmax(check)+24
        
    
    
def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 5)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
print(onset_samples)
librosa.display.waveshow(y=y, sr=sr)
plt.vlines(librosa.samples_to_time(onset_samples), -1, 1)
plt.show()

# number_of_segment = int(input())


segment = y[onset_samples[number_of_segment-1]:onset_samples[number_of_segment]]

print(str(fourier_pitch2(segment))+' to jest ten dźwięk')

