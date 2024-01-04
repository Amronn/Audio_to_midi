from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN, LSTM, Dropout, Conv1D, MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from keras.models import load_model

model = load_model('models/num_cqt.h5')

cqt_path = 'poliphonic_music/chords_cqt_2'
cqt = []
for filename in os.listdir(cqt_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(cqt_path, filename)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            chromas = list(reader)
        cqt.append(np.array(chromas, dtype=float))

cqt = np.array(cqt)
# print(cqt[0])

predictions = [model.predict(np.array([cqt[i]])) for i in range(10)]

print(predictions)