import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

df = pd.read_csv("Moons 2D Overlap.csv", header=None)


# for 2D df has 4 columns: (c0_x1, c0_x2, c1_x1, c1_2)
C0 = df[df.columns[:2]].values
C1 = df[df.columns[2:4]].values

X = np.empty((2 * len(df), 2), dtype=C0.dtype)
X[0::2] = C0          # even rows: class 0 points
X[1::2] = C1          # odd rows:  class 1 points

y = np.empty(2 * len(df), dtype=int)
y[0::2] = 0
y[1::2] = 1

# data is now stored in X unlabled sequentially
# X = [ [c0_1], [c1_1], ..., [c0_n], [c1_n]]
# y is array of labels [0, 1, ..., 0, 1]

# Splits data into 70% training, and then splits the remaining data in half for validation, and testing

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)



model = tf.keras.Sequential([
tf.keras.layers.Dense(12, activation='relu'), # 1st inner layer, 8 neurons, relu as activation function
tf.keras.layers.Dense(12, activation='relu'), # 2nd inner layer, 8 neurons, relu as activation function
tf.keras.layers.Dense(1, activation='sigmoid') # output layer, 1 neuron for binary classification
])

# define learning rate as 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])



history = model.fit(X_train, y_train, batch_size=32, epochs=150, validation_data=(X_valid, y_valid))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
model.evaluate(X_test, y_test)

# make sure to credit https://www.youtube.com/watch?v=VtRLrQ3Ev-U

# Plot points + ANN decision boundary
plt.figure(facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

plt.scatter(C0[:, 0], C0[:, 1], s=20, c="blue", alpha=0.8, label="Class 0")
plt.scatter(C1[:, 0], C1[:, 1], s=20, c="red", alpha=0.8, label="Class 1")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

probs = model.predict(grid, verbose=0).reshape(xx.shape)
plt.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Points + model decision boundary")
plt.legend()
plt.show()

# Plot training curves (accuracy / loss)
plt.figure(facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")

plt.xlabel("epoch")
plt.legend()
plt.title("Training loss and validation loss")
plt.show()


plt.figure(facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.xlabel("epoch")
plt.legend()
plt.title("Training accuracy and validation accuracy")
plt.show()


