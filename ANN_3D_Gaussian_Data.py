import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Remove '#' from dataset you would like to build ANN for
#df = pd.read_csv("Gaussian 3D Wide.csv", header=None)
#df = pd.read_csv("Gaussian 3D Narrow.csv", header=None)
#df = pd.read_csv("Gaussian 3D Overlap.csv", header=None)



# for 2D df has 4 columns: (c0_x1, c0_x2, c1_x1, c1_2)
C0 = df[df.columns[:3]].values
C1 = df[df.columns[3:6]].values

X = np.empty((2 * len(df), 3), dtype=C0.dtype)
X[0::2] = C0          # even rows: class 0 points
X[1::2] = C1          # odd rows:  class 1 points

y = np.empty(2 * len(df), dtype=int)
y[0::2] = 0
y[1::2] = 1

# data is now stored in X unlabled sequentially
# X = [ [c0_1], [c1_1], ..., [c0_n], [c1_n]]
# y is array of labels [0, 1, ..., 0, 1]

# Split data into 70% training, and then split the remaining data in half for validation, and testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)



model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(3,)),
tf.keras.layers.Dense(12, activation='relu'), # 1st inner layer, 12 neurons, relu as activation function
tf.keras.layers.Dense(12, activation='relu'), # 2nd inner layer, 12 neurons, relu as activation function
tf.keras.layers.Dense(1, activation='sigmoid') # output layer, 1 neuron for binary classification
])

# define learning rate as 0.005
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])



history = model.fit(X_train, y_train, batch_size=32, epochs=150, validation_data=(X_valid, y_valid))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
model.evaluate(X_test, y_test)

plt.figure(facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")

plt.xlabel("epoch")
plt.legend()
plt.title("Training history")
plt.show()

# make sure to credit https://www.youtube.com/watch?v=VtRLrQ3Ev-U