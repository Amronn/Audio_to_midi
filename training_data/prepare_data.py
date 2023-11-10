import numpy as np, librosa, os, csv
from sklearn.preprocessing import MinMaxScaler
file_name = 'training_data/88_notes'
number_of_keys = 88
number_of_samples = 4
data, sr = librosa.load(file_name)

oenv = librosa.onset.onset_strength(y=data, sr=sr, aggregate=np.mean, detrend=True)
oenv[oenv < (np.max(oenv) / 10)] = 0
onset_samples = librosa.onset.onset_detect(y=data, sr=sr, onset_envelope=oenv, backtrack=True, units='samples')
segment_size = int(sr * 0.05)
segments = np.array([data[i:i + segment_size] for i in onset_samples])

note_names = ['a','as','b','c', 'cs', 'd', 'ds', 'e', 'f', 'fs','g','gs']


# if number_of_samples*number_of_keys <= len(segments):
# for i in range(number_of_keys):    
#     folder_path = f'notes_88/{note_names[i%12]}{i//12}'
#     os.makedirs(folder_path, exist_ok=True)
#     for k in range(number_of_samples):
#         mel_spec = librosa.feature.melspectrogram(y=segments[i*number_of_samples+k], sr=sr, hop_length=256, n_mels = 128, n_fft=512)
#         # mfcc = librosa.feature.mfcc(y=segments[i*number_of_samples+k], sr=sr, n_mfcc=12, n_fft = 512)
#         scaler = MinMaxScaler()
#         mfcc = scaler.fit_transform(mel_spec)
#         f_name = f'mfcc_{k}.csv'
#         file_path = os.path.join(folder_path, f_name)
#         with open(file_path, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerows(mel_spec)
# else:
    # print('Niepoprawna wartosc number_of_keys i number_of_samples')

for i in range(number_of_keys):    
    folder_path = f'notes_88/{i+21}'
    os.makedirs(folder_path, exist_ok=True)
    for k in range(number_of_samples):
        mel_spec = librosa.feature.melspectrogram(y=segments[i*number_of_samples+k], sr=sr, hop_length=256, n_mels = 128, n_fft=512)
        # mfcc = librosa.feature.mfcc(y=segments[i*number_of_samples+k], sr=sr, n_mfcc=12, n_fft = 512)
        scaler = MinMaxScaler()
        mfcc = scaler.fit_transform(mel_spec)
        f_name = f'mfcc_{note_names[i%12]}{i//12}{k}.csv'
        file_path = os.path.join(folder_path, f_name)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(mel_spec)