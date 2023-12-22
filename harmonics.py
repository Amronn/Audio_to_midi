import numpy as np, librosa
file_name = ['liszt_frag.wav','bach.mp3', '88notes.wav']
file_path = 'wav_sounds/'+file_name[1]
y, sr = librosa.load(file_path)

f0, voicing, voicing_p = librosa.pyin(y=y, sr=sr, fmin=16, fmax=8192)
S = np.abs(librosa.stft(y))
freqs = librosa.fft_frequencies(sr=sr)
harmonics = np.array([1])
f0_harm = librosa.f0_harmonics(S, freqs=freqs, f0=f0, harmonics= harmonics)
import matplotlib.pyplot as plt
fig, ax =plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         x_axis='time', y_axis='log', ax=ax[0])
times = librosa.times_like(f0)
for h in harmonics:
    ax[0].plot(times, h * f0, label=f"{h}*f0")
ax[0].legend(ncols=4, loc='lower right')
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(f0_harm, ref=np.max),
                         x_axis='time', ax=ax[1])
ax[1].set_yticks(harmonics-1)
ax[1].set_yticklabels(harmonics)
ax[1].set(ylabel='Harmonics')
plt.show()