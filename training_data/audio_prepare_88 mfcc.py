import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
number_of_keys = 85
number_of_samples = 4

note_names = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs','g','gs','a','as','b']

file_path = 'E:\Amron\music_processing\project_audio_to_midi\piano_single_notes_labs.wav'
hop_length = 256
y, sr = librosa.load(file_path)
y = librosa.effects.harmonic(y)
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=12*3*7, hop_length=hop_length, filter_scale=0.6, sparsity=0.05))
threshold = 0.1
chroma_orig = librosa.feature.chroma_cqt(C=C, sr=sr, n_chroma=85, bins_per_octave=85*3, threshold=threshold, hop_length=hop_length, norm=1)

def get_onsets(y, sr):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, detrend=True)
    oenv[oenv < (np.max(oenv) / 20)] = 0
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, backtrack=True, units='samples').astype(int)
    onset_samples = np.concatenate([onset_samples, np.array([len(y)-1])])
    return onset_samples

onset_samples = get_onsets(y, sr)
onset_samples = [0]
for i in range(340):
    onset_samples.append(int(i*sr))
segment_size = int(sr * 0.05)
segments = np.array([y[i:i + segment_size] for i in onset_samples[:len(onset_samples)-1]])
mels = []
for seg in segments:
    mel_spect = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=2048, hop_length=256)
    mel_spect = np.mean(mel_spect, axis = 1)
    mels.append(mel_spect)
    # print(mel_spect.shape)


for i in range(number_of_keys):    
    folder_path = f'Audio_to_midi/notes_88_mels/{i}'
    os.makedirs(folder_path, exist_ok=True)
    for k in range(number_of_samples):
        f_name = f'mel_{note_names[i%12]}{i//12}{k}.csv'
        file_path = os.path.join(folder_path, f_name)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            data = mels[i*4+k]
            writer.writerows(map(lambda x: [x],data))
