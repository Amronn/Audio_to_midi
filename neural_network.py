from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


folder_path = 'Audio_to_midi/notes_88_cqt'
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
            y_data.append(int(note_label))

x_data = np.array(x_data).astype(float)
y_data = np.array(y_data)
# print(training_labels)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

print(y_train[2])
print(X_train[0].shape)

model = Sequential()

model.add(SimpleRNN(units = 85, input_shape=(1,85)))
print(model.output_shape)
model.add(Dense(170, activation='relu'))
print(model.output_shape)
model.add(Dense(85, activation='softmax'))
print(model.output_shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

prediction = model.predict(np.array([X_train[2]]))

# print(prediction)

model.save('Audio_to_midi/notes_88_cqt.h5')
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

print(np.argmax(prediction))
print(np.max(prediction))