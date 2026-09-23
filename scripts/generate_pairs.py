import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "pose_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "pose_pairs_dataset.csv")
RANDOM_SEED = 42

# Load dataset
df = pd.read_csv(DATASET_PATH)
rng = np.random.default_rng(RANDOM_SEED)

positive_pairs = []
negative_pairs = []

poses = df["pose"].unique()
pose_groups = {
    pose: df[df["pose"] == pose].reset_index(drop=True)
    for pose in poses
}

for pose in poses:

    pose_rows = pose_groups[pose]

    # positive pairs (same pose)
    for i in range(len(pose_rows)):
        for j in range(i+1, len(pose_rows)):

            a = pose_rows.iloc[i].drop("pose").values
            b = pose_rows.iloc[j].drop("pose").values

            positive_pairs.append(list(a) + list(b) + [1])

positive_count = len(positive_pairs)

for _ in range(positive_count):

    pose_a, pose_b = rng.choice(poses, size=2, replace=False)
    row_a = pose_groups[pose_a].iloc[rng.integers(len(pose_groups[pose_a]))]
    row_b = pose_groups[pose_b].iloc[rng.integers(len(pose_groups[pose_b]))]

    negative_pairs.append(
        list(row_a.drop("pose").values) +
        list(row_b.drop("pose").values) +
        [0]
    )

pairs = positive_pairs + negative_pairs

# convert to dataframe
feature_count = len(df.columns) - 1

columns = []

for i in range(feature_count):
    columns.append(f"a{i}")

for i in range(feature_count):
    columns.append(f"b{i}")

columns.append("label")

pair_df = pd.DataFrame(pairs, columns=columns)
pair_df = pair_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

pair_df.to_csv(OUTPUT_PATH, index=False)

print("Pair dataset created: pose_pairs_dataset.csv")
print("Total pairs:", len(pair_df))
print("Label counts:")
print(pair_df["label"].value_counts().sort_index().to_string())
