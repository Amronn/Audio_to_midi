import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from mido import Message, MidiFile, MidiTrack
import fun
file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav']
file_path = 'wav_sounds/'+file_name[0]

hop_length = 256
y, sr = librosa.load(file_path)

chroma_orig = fun.chroma(hop_length, y, sr, 85)

''' jakas dziwna moja metoda nie wyszła
def correct_pitch(chroma, threshold=0.1, wiener_size=3):
    chroma = scipy.signal.wiener(chroma, wiener_size)
    threshold = 0.1
    chroma[chroma<threshold]=0
    peaks, what = scipy.signal.find_peaks(chroma)
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
'''
def avr_w(data):
    n = len(data)
    weights = [np.exp(-i/10)*10 for i in range(1, n+1)]
    average_with_weights = sum(w * x for w, x in zip(weights, data)) / sum(weights)
    return average_with_weights
def alikwot_check2(chroma, num_of_harmonics = 3):

    chroma2 = scipy.signal.wiener(chroma, 3)
    threshold = 0.05
    chroma2[chroma2<threshold]=0
    peaks, what = scipy.signal.find_peaks(chroma2)

    # plt.plot(chroma2)
    # plt.show()
    #korekta wysokości w zależności od wpływu harmonicznych. Zakładam, że dźwiękiem podstawowym jest pierwsza harmoniczna.
    if len(peaks)==1:
        if peaks[0]!=None:
            return int(peaks[0])+24
        else:
            return 0
    check = 0
    if len(peaks)>1:
        har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
        harmoniczne = np.zeros(85)
        k=0
        for i in har[:4]:
            harmoniczne[i] = 1 + np.exp(-2*k)
            k=k+1
        
        x = range(0,85)
        # plt.plot(x, harmoniczne, 'o', label='Dane punkty')
        # plt.legend()
        # plt.show()
        check = np.correlate(chroma2, harmoniczne, 'full')
        # plt.plot(check[84:])
        # plt.show()
        if np.argmax(check[84:])+24 != None:
            return int(np.argmax(check[84:])+24)
        else:
            return 0
    # return np.argmax(chroma2)+24
def alikwot_check(chroma):
    chroma /= np.max(chroma)
    har = [0, 12, 19, 24]
    harmoniczne = np.zeros(85)

    cor = np.zeros(85)
    for n in range(1,len(har)):
        harmoniczne = np.zeros_like(cor)
        for i in har[:n]:
            harmoniczne[i] = 1
        cor += np.correlate(chroma, harmoniczne, 'full')[84:]

    return np.argmax(cor)+24

onset_samples = fun.get_onsets(y, sr)
onset_samples = np.unique(onset_samples)

librosa.display.specshow(chroma_orig, y_axis='chroma', x_axis='time', sr=sr)
plt.grid()
plt.show()

plt.figure(figsize=(15,6))
librosa.display.waveshow(y,sr=sr)
plt.vlines(librosa.samples_to_time(onset_samples), ymin=-1, ymax=1)
plt.show()


onset_samples_cqt = (onset_samples/hop_length).astype(int)
print(onset_samples_cqt)
chroma=[]
for i in range(0,len(onset_samples_cqt)-1):
    chroma.append(chroma_orig[:, onset_samples_cqt[i]:onset_samples_cqt[i+1]])

chroma_av = []
i=0
for chroma in chroma:
    chromas = []
    for ch in chroma:
        chromas.append(avr_w(ch))
    i=i+1
    chroma_av.append(chromas)

pitches_list = []
for chroma_av in chroma_av:
    pitches_list.append(alikwot_check2(chroma_av))
# print(len(pitches_list))
print(pitches_list)
timesx = librosa.samples_to_time(onset_samples)
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

def create_midi(pitches_list, times):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for i, time in enumerate(times):
        track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
        track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

    mid.save('monophonic_music/audio_to_midi/cqt_chroma_midi.mid')

create_midi(pitches_list, times)

