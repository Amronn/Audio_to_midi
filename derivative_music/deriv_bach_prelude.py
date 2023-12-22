import numpy as np, librosa, os, csv
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from mido import Message, MidiFile, MidiTrack
import sounddevice as sd

file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav']
# file_path = 'wav_sounds/'+file_name[0]
file_path = 'wav_sounds/liszt_frag.wav'
hop_length = 256
y, sr = librosa.load(file_path)

print(len(y))

frag = [0, 4, 7, 12, 16, 7, 12, 16, 0, 4, 7, 12, 16, 7, 12, 16,0, 2, 9, 14, 17, 9, 14, 17, 0, 2, 9, 14, 17, 9, 14, 17]


x = range(len(frag))

plt.figure(figsize=(10, 5))
plt.subplot(121),plt.plot(x, frag)

new_frag = np.zeros_like(frag)
new_frag2 = np.zeros_like(frag)
new_y = np.zeros_like(y)
new_y2 = np.zeros_like(y)

for i in range(len(frag)-1):
    new_frag[i] = librosa.midi_to_hz(frag[i+1]+60) - librosa.midi_to_hz(frag[i]+60)
    
for i in range(len(y)-1):
    new_y[i] = y[i+1] - y[i]
for i in range(len(y)-1):
    new_y2[i] = new_y[i+1] - new_y[i]
# sd.play(data = new_y2, samplerate=sr)
# sd.wait()
    
plt.subplot(122),plt.plot(x, new_frag)
# plt.show()
pitches_list = librosa.hz_to_midi(new_frag).astype(int)
times = np.ones_like(pitches_list)
new_frag = librosa.midi_to_note(new_frag+60)
print(new_frag)
from mido import Message, MidiFile, MidiTrack

def create_midi(pitches_list, times):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for i, time in enumerate(times):
        track.append(Message('note_on', note=pitches_list[i], velocity=127, time = 0))
        track.append(Message('note_off', note=pitches_list[i], velocity=127, time = time))

    mid.save('derivative_music/derivative_bach.mid')

create_midi(pitches_list, times*200)