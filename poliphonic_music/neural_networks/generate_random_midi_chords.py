import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

number = 1000

times = [np.random.randint(1000,2000) for i in range(number)]
chords = []

for k in range(number):
    n = np.random.randint(1,5)
    chord = []
    for i in range(n):
        chord.append(np.random.randint(40,80))
    chord = np.unique(chord)
    chords.append(chord)


from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for i, time in enumerate(times):
    # track.append(Message('note_on', note=chords[i], velocity=127, time = 1000))
    # track.append(Message('note_off', note=chords[i], velocity=127, time = 0))
    for note in chords[i]:
        track.append(Message('note_on', note=note, velocity=127, time = 0))
    for note in chords[i]:
        if note == chords[i][0]:
            track.append(Message('note_off', note=note, velocity=127, time = time))
        else:
            track.append(Message('note_off', note=note, velocity=127, time = 0))
    track.append(Message('note_off', note=note, velocity=127, time = 500))
mid.save('poliphonic_music/test2.mid')

csv_file_path = 'poliphonic_music/chord_data2.csv'

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'Chords'])

    for i in range(number):
        writer.writerow([times[i], *chords[i]])
        print(i)