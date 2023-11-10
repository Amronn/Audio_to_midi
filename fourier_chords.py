import numpy as np
import scipy.signal
import scipy.ndimage
import librosa
# file_path = 'wav_sounds/piano_sample_2.wav'
file_path = 'wav_sounds/liszt_frag.wav'

x, sr = librosa.load(file_path)

def fourier_pitch(segment, sr=sr, fmin=16, fmax=8192):
    X = np.abs(np.fft.fft(segment, fmax-fmin))
    X = X[int(fmin * len(X) / sr):int(fmax * len(X) / sr)]
    f = np.linspace(fmin, fmax, len(X))
    note = np.rint(librosa.hz_to_midi(f[np.argmax(X)])).astype(int)
    return note

oenv = librosa.onset.onset_strength(y=x, sr=sr, aggregate=np.mean, detrend=True)
oenv[oenv < 0.1*np.max(oenv) ] = 0
onset_samples = librosa.onset.onset_detect(y=x, sr=sr, onset_envelope=oenv, backtrack=True, units='samples')
segment_size = int(sr * 0.2)
segments = np.array([x[i:i + segment_size] for i in onset_samples])

all_hits = np.concatenate([onset_samples, np.array([len(x) - 1])])


'''


import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
librosa.display.waveshow(x,sr=sr)
plt.vlines(librosa.samples_to_time(all_hits), ymin=-1, ymax=1, color='red')
plt.show()

'''



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