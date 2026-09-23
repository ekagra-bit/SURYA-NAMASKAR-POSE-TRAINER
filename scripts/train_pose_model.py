import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

tf.get_logger().setLevel("ERROR")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_DATASET = os.path.join(BASE_DIR, "pose_pairs_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "pose_embedding_model.keras")
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2


def stratified_split_indices(labels, validation_split, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)

        val_count = max(1, int(len(label_indices) * validation_split))
        if val_count >= len(label_indices):
            val_count = len(label_indices) - 1

        val_indices.extend(label_indices[:val_count])
        train_indices.extend(label_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices), np.array(val_indices)

df = pd.read_csv(PAIR_DATASET)

X = df.drop("label", axis=1).values
X = X.astype("float32")
max_value = X.max()
if max_value > 0:
    X = X / max_value
y = df["label"].values

feature_count = int(X.shape[1] / 2)
train_indices, val_indices = stratified_split_indices(
    y,
    validation_split=VALIDATION_SPLIT,
    seed=RANDOM_SEED
)

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]

Xa_train = X_train[:, :feature_count]
Xb_train = X_train[:, feature_count:]
Xa_val = X_val[:, :feature_count]
Xb_val = X_val[:, feature_count:]

# embedding network
def create_embedding_model():

    inp = layers.Input(shape=(feature_count,))

    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8)(x)

    return models.Model(inp, x)

embedding_model = create_embedding_model()

inputA = layers.Input(shape=(feature_count,))
inputB = layers.Input(shape=(feature_count,))

embA = embedding_model(inputA)
embB = embedding_model(inputB)

class EuclideanDistance(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)

distance = EuclideanDistance()([embA, embB])

output = layers.Dense(1, activation="sigmoid")(distance)

model = models.Model(inputs=[inputA, inputB], outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("Train label counts:")
print(pd.Series(y_train).value_counts().sort_index().to_string())
print("Validation label counts:")
print(pd.Series(y_val).value_counts().sort_index().to_string())

model.fit(
    [Xa_train, Xb_train],
    y_train,
    batch_size=32,
    epochs=8,
    validation_data=([Xa_val, Xb_val], y_val),
    verbose=2
)

model.save(MODEL_PATH)
model.save("saved_model", save_format="tf")

import os
print("Saved at:", os.path.abspath(MODEL_PATH))
