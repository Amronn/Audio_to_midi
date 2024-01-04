import numpy as np
import librosa
import matplotlib.pyplot as plt

#to nie działa, bo nie jest zrobione

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav']
file_path = 'wav_sounds/'+file_name[0]
y, sr = librosa.load(file_path)

oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=False)
oenv[oenv < (np.max(oenv) / 10)] = 0
onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=False, units='samples').astype(int)
onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
segment_size = int(sr * 0.1)
segments = np.array([y[i:i + segment_size] for i in onset_samples])

all_hits = np.concatenate([onset_samples, np.array([len(y) - 1])])

plt.figure(figsize=(15,6))
librosa.display.waveshow(y,sr=sr)
plt.vlines(librosa.samples_to_time(all_hits), ymin=-1, ymax=1)
plt.show()

mel_spect = librosa.feature.melspectrogram(y=y[0:20000], sr=sr, n_fft=2048, hop_length=256)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', x_axis='time')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()

pitches_list = []

for segment in segments:
    pitch = 0
    pitches_list.append(pitch)

# print(pitches_list)

timesx = librosa.samples_to_time(all_hits)
times = list(int((timesx[i+1]-timesx[i])*1000) for i in range(len(timesx)-1))

# print(times)


from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0)) # velocity ustawione jak narazie na stałe
    print(time)
    track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time)) # tutaj nie ma znaczenia

mid.save('monophonic_music/audio_to_midi/mfcc_audio_to_midi.mid')
