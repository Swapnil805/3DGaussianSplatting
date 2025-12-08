import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np

input_folder = "../data/frames/test"
output_folder = "../data/frames/test_blurry"

os.makedirs(output_folder, exist_ok=True)

def variance_of_laplacian(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

# Step 1: compute sharpness scores
scores = []
images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
for img_name in tqdm(images, desc="Scanning frames"):
    path = os.path.join(input_folder, img_name)
    score = variance_of_laplacian(path)
    scores.append(score)

# Step 2: pick adaptive threshold
scores = np.array(scores)
median_score = np.median(scores)
THRESHOLD = max(5, 0.3 * median_score)  # more lenient baseline

print(f"\n📊 Median sharpness: {median_score:.2f}")
print(f"🧠 Using adaptive threshold: {THRESHOLD:.2f}")

# Step 3: move only very blurry frames
kept, removed = 0, 0
for img_name, score in zip(images, scores):
    src = os.path.join(input_folder, img_name)
    if score < THRESHOLD:
        shutil.move(src, os.path.join(output_folder, img_name))
        removed += 1
    else:
        kept += 1

print(f"\n✅ Done! Kept {kept} sharp frames, moved {removed} blurry ones → {output_folder}")


# TO CHECK SHARPNESS
# import cv2
# import glob
# import numpy as np
#
# paths = sorted(glob.glob("../data/frames/test/*.png"))[:10]  # check first 10
# for p in paths:
#     img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#     score = cv2.Laplacian(img, cv2.CV_64F).var()
#     print(f"{p.split('/')[-1]}: {score:.2f}")
