import numpy as np
import scipy.signal
import scipy.ndimage
import librosa
import matplotlib.pyplot as plt
# file_path = 'wav_sounds/piano_sample_2.wav'
# file_path = 'wav_sounds/liszt_frag.wav'
file_path = 'wav_sounds/four_chords.wav'
x, sr = librosa.load(file_path)

num = 1
    
def fourier_pitch(segment, sr=sr, num_of_harmonics = 10, fmin=16, fmax=8192):
    N = len(segment)
    print(N)
    X = np.abs(np.fft.fft(segment))
    print(len(X))
    f = np.linspace(0, sr, len(X))

    plt.plot(f, X)
    plt.show()
    f0 = 32.70
    a = 2**(1/12)
    notes = [f0*a**n for n in range(85)]
    for note in notes:
        harmoniczne = [k*note for k in range(1,num_of_harmonics+1) if k*note<len(X)]
        x = np.linspace(0,N/sr,N)
        signal = 0
        if len(harmoniczne)>0:
            for h in harmoniczne:
                signal = signal+np.cos(2*np.pi*h*x)
            # plt.plot(x, signal)
            # plt.show()
            X2 = np.abs(np.fft.fft(signal))
            X2 /= np.max(X2)
            X2[X2>0.1] = 1
            X2[X2<=0.1] = 0
            f = np.linspace(0, sr, len(X2))
            # plt.plot(f, X2)
            
def fourier_pitch2(segment, sr=sr, num_of_harmonics = 10, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    avr = 0
    while avr<np.max(X)/5:
        avr = np.mean(X[X>0])
        X[X<avr] = 0
    first = 0
    for i in range(len(X)-1):
        if X[i+1]>X[i]:
            first = i+1
            break
    print(f[first])
    # avr = np.ones_like(f)*avr
    # plt.plot(f, X)
    # plt.plot(f, avr)
    # plt.show()
    return f[first]

oenv = librosa.onset.onset_strength(y=x, sr=sr, aggregate=np.mean, detrend=True)
oenv[oenv < 0.1*np.max(oenv) ] = 0
onset_samples = librosa.onset.onset_detect(y=x, sr=sr, onset_envelope=oenv, backtrack=True, units='samples')
segment_size = int(sr * 0.2)
segments = np.array([x[i:i + segment_size] for i in onset_samples])

all_hits = np.concatenate([onset_samples, np.array([len(x) - 1])])

segment = segments[num-1]

fourier_pitch(segment)

'''


import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
librosa.display.waveshow(x,sr=sr)
plt.vlines(librosa.samples_to_time(all_hits), ymin=-1, ymax=1, color='red')
plt.show()


pitches_list = []

for segment in segments:
    pitch = fourier_pitch(segment)
    pitches_list.append(pitch)

print(pitches_list)

timesx = librosa.samples_to_time(all_hits)
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

print(times)


from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0)) # velocity ustawione jak narazie na stałe
    print(time)
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time)) # tutaj nie ma znaczenia

mid.save('fourier_audio_to_midi.mid')

'''