from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN, LSTM, Dropout, Conv1D
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam


etykiety = []
dane = []

for i in range(200):
    a = [np.random.randint(0,2) for k in range(10)]
    etykiety.append(a)
    b = []
    for a1 in a:
        b.append(a1+np.random.randn()/10)
    dane.append(b)
    
etykiety = np.array(etykiety)
dane = np.array(dane)
print(etykiety[0])
print(dane[0])

X_train, X_test, y_train, y_test = train_test_split(dane, etykiety, test_size=0.4, random_state=43)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

etykiety = []
dane = []

for i in range(1000):
    a = [np.random.randint(0,2) for k in range(10)]
    etykiety.append(a)
    b = []
    for a1 in a:
        b.append(a1+np.random.randn()/20)
    dane.append(b)
etykiety = np.array(etykiety)
dane = np.array(dane)

def new_func(etykiety, dane, k):
    tab = []
    prediction = model.predict(np.array([dane[k]]))
    for i in range(10):
        if etykiety[k][i]==1:
            tab.append(i)
    plt.plot(prediction[0])
    plt.vlines(tab, ymin=0, ymax=1, color='red')
    plt.show()
# for k in range(10):
#     new_func(y_test, X_test, k=k)
    
for k in range(10):
    new_func(etykiety, dane, k=k)