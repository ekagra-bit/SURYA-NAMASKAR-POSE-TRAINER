import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel("ERROR")

# MoveNet model load
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures["serving_default"]

DATASET = []

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_FOLDER = os.path.join(BASE_DIR, "reference")
OUTPUT_PATH = os.path.join(BASE_DIR, "pose_dataset.csv")

def detect_pose(image):
    img = tf.image.resize_with_pad(image, 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    outputs = movenet(img)
    keypoints = outputs["output_0"].numpy()[0][0]
    return keypoints

for pose_name in os.listdir(REFERENCE_FOLDER):

    pose_path = os.path.join(REFERENCE_FOLDER, pose_name)

    if not os.path.isdir(pose_path):
        continue

    for img_name in os.listdir(pose_path):

        img_path = os.path.join(pose_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = tf.expand_dims(img, axis=0)

        keypoints = detect_pose(img_tensor)

        row = [pose_name]

        for kp in keypoints:
            row.append(kp[1])  # x
            row.append(kp[0])  # y

        DATASET.append(row)

columns = ["pose"]

for i in range(17):
    columns.append(f"x{i}")
    columns.append(f"y{i}")

df = pd.DataFrame(DATASET, columns=columns)

df.to_csv(OUTPUT_PATH, index=False)

print("Dataset saved:", OUTPUT_PATH)
