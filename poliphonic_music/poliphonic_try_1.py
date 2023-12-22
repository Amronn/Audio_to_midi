import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from mido import Message, MidiFile, MidiTrack

# file_path = 'wav_sounds/piano_2_notes.wav'
# file_path = 'wav_sounds\piano_test2.wav'
# file_path = 'wav_sounds\piano_chords_123.wav'
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

#korelacja z kolejnymi harmonicznymi
def alikwot_check(chroma, num_of_harmonics = 3):
    # chroma = np.convolve(chroma, np.array([5,5]))
    chroma = np.array(chroma)
    plt.plot(chroma)
    plt.show()
    # chroma[chroma>0.1] = 1
    # chroma[chroma<=0.1] = 0
    # chroma2 = scipy.signal.wiener(chroma, 3)
    # threshold = 0.1
    # chroma2[chroma2<threshold]=0
    peaks, what = scipy.signal.find_peaks(chroma)
    
    chroma = scipy.signal.wiener(chroma, 3)
    x = range(85)
    plt.plot(chroma)
    plt.show()
    
    chroma = scipy.signal.wiener(chroma, 3)
    x = range(85)
    plt.plot(chroma)
    plt.show()
    # chroma2[chroma2>0.1] = 1
    # chroma2[chroma2<0.1] = 0
    # plt.plot(chroma2)
    # plt.show()
    # print(chroma2)
    if len(peaks)==1:
        return int(peaks[0])+24
    check = 0
    if len(peaks)>1:
        har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
        harmoniczne = np.zeros(85)
        k=0
        for i in har[:4]:
            harmoniczne[i] = 1 + np.exp(-2*k)
            k=k+1

        check = np.correlate(chroma, harmoniczne, 'full')
        check = check[84:]
        peaks, what = scipy.signal.find_peaks(check)
        
        plt.plot(check)
        plt.show()
        print(peaks)
        
        number_of_max = 0
        
        idx_to_pop = np.argmax(check)
        
        for p in peaks:
            if check[p] == max(check):
                number_of_max +=1
                
        new_peaks = []    
        if number_of_max == 1:
            for peak in peaks:
                if peak!=idx_to_pop:
                    new_peaks.append(peak)
        else:
            new_peaks = peaks
        maks = np.max(check[new_peaks])
        notes_idx = []
        for peak in new_peaks:
            if check[peak] == maks:
                notes_idx.append(peak+24)
                
                
        print(new_peaks)
        plt.plot(check[new_peaks])
        plt.show()
        
        return notes_idx

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 10)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
onset_samples = np.unique(onset_samples)


onset_samples_cqt = (onset_samples/hop_length).astype(int)
print(onset_samples_cqt)
chroma = chroma_orig[:, onset_samples_cqt[num-1]:onset_samples_cqt[num]]

librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
plt.grid()
plt.show()
chr = []
for ch in chroma:
    chr.append(avr_w(ch))
chroma = chr
# chroma = (np.mean(chroma, axis = 1))

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
