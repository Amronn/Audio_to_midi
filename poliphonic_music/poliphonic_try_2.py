import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from mido import Message, MidiFile, MidiTrack

file_path = 'wav_sounds/piano_chords_scale_in_C.wav'
hop_length = 256
y, sr = librosa.load(file_path)

num = 1

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length))
threshold = 0.0
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length)
chroma_orig = scipy.signal.convolve2d(chroma_orig, np.ones((1,3))/10)

def avr_w(data):
    n = len(data)
    weights = [np.exp(-i/10)*10 for i in range(1, n+1)]
    average_with_weights = sum(w * x for w, x in zip(weights, data)) / sum(weights)
    return average_with_weights


def alikwot_check(chroma, num_of_harmonics = 6):
    har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
    harmoniczne = np.zeros(85)
    k=0
    for i in har[:num_of_harmonics]:
        harmoniczne[i] = 1 + np.exp(-2*k)
        k=k+1
    cor = np.correlate(chroma, harmoniczne,'full')[84:]
    plt.plot(cor)
    plt.show()
    
def fourier_check(segment, fmin = 16, fmax = 8192):
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    plt.plot(f, X)
    plt.show()
    

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
onset_samples = np.unique(onset_samples)


onset_samples_cqt = (onset_samples/hop_length).astype(int)

chroma = chroma_orig[:, onset_samples_cqt[num-1]:onset_samples_cqt[num]]

librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
plt.grid()
plt.show()
chr = []
for ch in chroma:
    chr.append(avr_w(ch))
chroma = chr

fourier_check(segment=y[onset_samples[num-1]:onset_samples[num]])
print(alikwot_check(chroma))

'''

chromas = []
onset_samples_cqt = (onset_samples/hop_length).astype(int)
# print(onset_samples_cqt)
for i in range(len(onset_samples_cqt)-1):
    chroma = chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]]
    chromas.append(chroma)
    


chroma_av = []
for chroma in chromas:
    # chromas_av.append(avr_w(chroma))
    chroma_av.append(np.mean(chroma, axis = 1))


# plt.figure(figsize=(15,6))
# librosa.display.waveshow(y,sr=sr)
# plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1)
# plt.show()
    # print(librosa.midi_to_note(np.argmax(np.mean(chroma, axis=1))+24))
pitches_list = []
for ch_av in chroma_av:
    pitches_list.append(alikwot_check(ch_av))
    

timesx = librosa.samples_to_time(onset_samples)
timesx = list(int(time*1000) for time in timesx)
# print(timesx)
times = list(int((timesx[i+1]-timesx[i])) for i in range(len(timesx)-1))
# print(times)
def create_midi(pitches_list, times):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for i in range(len(pitches_list)):
        chord_pitches = pitches_list[i]
        chord_time = times[i]
        # print(chord_time)

        for pitch in chord_pitches:
            track.append(Message('note_on', note=pitch, velocity=127, time=0))

        for pitch in chord_pitches:
            track.append(Message('note_off', note=pitch, velocity=127, time=chord_time))

    mid.save('poli_midi.mid')

create_midi(pitches_list, times)

'''
