import numpy as np
import librosa

f0 = 32.70
a = 2**(1/12)
notes = [f0*a**n for n in range(85)]
num_har = 8
alikwoty = []
for i in range(85):
    # alikwoty.append(librosa.hz_to_midi([notes[i]*n for n in range(1,num_har+1)]).astype(int))
    alikwoty.append(librosa.hz_to_midi([notes[i]*n for n in range(1,num_har+1)]))

print(alikwoty)
while True:
    note = int(input("wpisz nute: "))
    for i, al in enumerate(alikwoty):
        for a in al:
            if a==note:
                print(librosa.midi_to_note(i+23))
    # print(note)