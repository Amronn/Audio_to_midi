import numpy as np
import random
akordy = ['maj7', 'mi7', 'polzmniejszony', 'D7']
tonacja = ['C', 'C#', 'Cb', 'D', 'D#', 'Db', 'E', 'E#', 'Eb', 'F', 'F#', 'Fb', 'G', 'G#', 'Gb', 'A', 'A#', 'Ab', 'B', 'B#', 'Bb']
przewroty = ['0', '1', '2', '3']

for i in range(10):
    r_a = random.randint(0,len(akordy)-1)
    r_t = random.randint(0, len(tonacja)-1)
    r_p = random.randint(0, len(przewroty)-1)
    if akordy[r_a] == 'polzmniejszony' and r_p != '0':
        r_p = '0'
    print(tonacja[r_t]+akordy[r_a])
    print("przewrót %s" %r_p +'\n') 