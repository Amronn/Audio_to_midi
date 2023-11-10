import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

l_prob = 4096
rate = 44100

frequencies = np.fft.rfftfreq(l_prob, d=1/rate)
magnitudes = np.zeros(len(frequencies))

def audio_callback(indata, frames, time, status):
    global magnitudes
    window = np.kaiser(len(indata[:, 0]),0) # ..., 0 prostokatne jednak XD
    windowed_signal = indata[:, 0] * window
    magnitudes = np.abs(np.fft.rfft(windowed_signal, n=l_prob))
    # magnitudes[magnitudes <= 5] = 0


fig, ax = plt.subplots()

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Real-time Audio Spectrum Analyzer')

line, = ax.plot([], [])
ax.set_xlim(0, 500)
ax.set_ylim(0, 100)

def update_plot(frame):
    line.set_data(frequencies, magnitudes)

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=rate)
stream.start()

ani = FuncAnimation(fig, update_plot, interval=10)

plt.show()

try:
    while True:
        plt.pause(0.01)
except KeyboardInterrupt:
    pass

stream.stop()
stream.close()
