import numpy as np
import librosa
import matplotlib.pyplot as plt

note_num = 60

N=10000
sr = 22500
fmin = 16
fmax = 8196
f0 = 32.70
a = 2**(1/12)
notes = [f0*a**n for n in range(85)]
note = notes[60-24]
harmoniczne = [k*note for k in range(1,10) if k*note<16000]
x = np.linspace(0,N/sr,N)

signal = 0
if len(harmoniczne)>0:
    for h in harmoniczne:
        signal = signal+np.cos(2*np.pi*h*x)

note = notes[67-24]
harmoniczne = [k*note for k in range(1,10) if k*note<16000]
signal1 = 0
if len(harmoniczne)>0:
    for h in harmoniczne:
        signal1 = signal1+np.cos(2*np.pi*h*x)
        
signal2 = signal+signal1
        
plt.plot(x, signal)
plt.show()

X2 = np.abs(np.fft.fft(signal2))
X2 /= np.max(X2)
f = np.linspace(0, sr, len(X2))
plt.plot(f[int(fmin/sr*N):int(fmax/sr*N)], X2[int(fmin/sr*N):int(fmax/sr*N)])
plt.show()