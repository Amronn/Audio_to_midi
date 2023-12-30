from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


folder_path = 'notes_88_v3_cqt'
x_data = []
y_data = []
for folder_name in os.listdir(folder_path):
    folder_dir = os.path.join(folder_path, folder_name)
    if not os.path.isdir(folder_dir):
        continue
    note_label = folder_name
    for filename in os.listdir(folder_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_dir, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                chromas = list(reader)
            x_data.append(np.array(chromas).T)
            # y_data.append(int(note_label))
            print(note_label)
            print(int(note_label)//12)
            y_data.append(int(note_label)//12)
                
            # print(note_label)

x_data = np.array(x_data).astype(float)
y_data = np.array(y_data)
print(y_data)
# print(x_data[34], y_data[34])

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

# print(y_train[2])
# print(X_train[0].shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(SimpleRNN(units = 85, input_shape=(1,85)))

model.add(Dense(48, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))

model.add(Dense(8, activation='softmax'))


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.01, amsgrad=True)

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[reduce_lr])
print(np.argmax(X_train[2]))
prediction = model.predict(np.array([X_train[2]]))

# print(prediction)

model.save('octaves_cqt.h5')
# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = np.arange(0, 85, 1)
# Y = np.arange(0, 85, 1)
# X, Y = np.meshgrid(X, Y)

# prediction = np.array(prediction)

# # Ensure that the shape of the matrix matches the dimensions of X and Y
# if prediction.shape == (len(X), len(Y)):
#     Z = prediction

#     # Plot the surface.
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)

#     ax.zaxis.set_major_locator(LinearLocator(10))

#     fig.colorbar(surf, shrink=0.5, aspect=5)

#     plt.show()
# else:
#     print("Shape of 'prediction' does not match the dimensions of X and Y.")
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

# Show the plots
plt.tight_layout()
plt.show()
print(np.argmax(prediction))
print(np.max(prediction))