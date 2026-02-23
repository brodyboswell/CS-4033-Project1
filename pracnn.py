import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

df = pd.read_csv("Gaussian 2D Narrow.csv", header=None)


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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)



model = tf.keras.Sequential([
tf.keras.layers.Dense(12, activation='relu'), # 1st inner layer 
tf.keras.layers.Dense(12, activation='relu'), # 2nd inner layer 
tf.keras.layers.Dense(1, activation='sigmoid') # output layer
])

# define learning rate as 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])



model.fit(X_train, y_train, batch_size=32, epochs=40, validation_data=(X_valid, y_valid))

model.evaluate(X_test, y_test)

# make sure to credit https://www.youtube.com/watch?v=VtRLrQ3Ev-U




