import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]] # random selection
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

# print the shapes of input dataset for understanding
print(f'Shape of X:', X_train.shape)
print(f'Shape of y:', y_train.shape)

# setup model
model = Sequential(
    [
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear'), # softmax activation fn will be used as we are setting from_legits=True
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # sending input as legits rather than probabilities            
    optimizer=tf.keras.optimizers.Adam(0.001)                             # for higher accuracy
)

model.fit(
    X_train, y_train,
    epochs=10
)

# this output is not probability but z value
output_z = model.predict(X_train)
print(f"two example output vectors:\n {output_z[:2]}")
print("largest value", np.max(output_z), "smallest value", np.min(output_z))

# process these z values using softmax function to get probabilities
output_probs = tf.nn.softmax(output_z).numpy()
print(f"two example output vectors:\n {output_probs[:2]}")
print("largest value", np.max(output_probs), "smallest value", np.min(output_probs))

for i in range(5):
    print( f"{output_z[i]}, category: {np.argmax(output_z[i])}")
