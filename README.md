# 3DGaussianSplatting : Create a 3D Gaussian Splatting scene of a scene

## 🎯 Project Overview
In this project, we are implementing a **visual computing pipeline** to reconstruct a 3D scene from a multi-view image sequence using **3D Gaussian Splatting**.

The system will transform 2D image frames into a **real-time renderable radiance field**, enabling smooth, interactive viewpoint navigation.

We are achieving this through these high level steps:

1. **Structure-from-Motion (SfM)** using **COLMAP**  
   → Recovers **camera poses** + **sparse 3D point cloud**

2. **Gaussian Splatting Initialization**  
   → Converts COLMAP reconstruction into anisotropic 3D Gaussians

3. **Gaussian Optimization + Rendering**  
   → Produces a photorealistic, real-time view-synthesis model

This work follows the method introduced in:  (base research paper)

> **Kerbl, Bernhard, et al.**  
> *"3D Gaussian Splatting for Real-Time Radiance Field Rendering."*  
> **ACM TOG 42.4 (2023).**

---

## 🚀 Project Layers

To de-risk development we split the work into five incremental layers:

• **Functional Minimum**: Generate a 3D point cloud from multi-view images using COLMAP
and visualize it.

• **Minimum Goal**: Convert COLMAP results to Gaussian representation and render a basic
static scene.

• **Desired Goal**: Achieve real-time rendering with adjustable camera viewpoints.

• **Maximum Goal**: Add basic user controls (camera orbit, zoom, reset) or improve the
rendering pipeline with automated parameter setup.

• **Extras**: Automate the full pipeline (video → frames → COLMAP → Gaussian → render).

This project demonstrates applied understanding of **computer vision, 3D geometry, GPU optimization, and rendering**.

---

# 📂 Environment Setup & Pipeline Steps

Below is the exact pipeline followed by us to reach the interim goal of our project

---

## 1. System level environment setup

We are using MacOS as our base OS, so we first installed **Homebrew** (macOS package manager)  
  Install command used:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # then added brew to PATH:
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  eval "$(/opt/homebrew/bin/brew shellenv)"
  ```
Then we installed the required tools using brew commands : 
```bash
brew install ffmpeg
brew install colmap
```
After this, we created the python virtual environment and installed the basic libraries which we were going to use in our project : (used PyCharm Community edition)
```bash
python3 -m venv gs_env
source gs_env/bin/activate
pip install --upgrade pip
```
Installing libraries through pycharm terminal:
```bash
pip install torch torchvision torchaudio      # PyTorch (Apple MPS locally)
pip install open3d numpy matplotlib opencv-python pillow tqdm
pip install jupyterlab 
```
 ### Project File Structure:
 ```bash
GaussianSplatting_Project/
 ├─ data/
 │   ├─ raw/                     # original video(s)
 │   ├─ frames/test/             # extracted frames used for COLMAP
 │   └─ colmap_output/           # COLMAP outputs (database, sparse, etc.)
 ├─ scripts/
 │   ├─ frame_extraction.py
 │   ├─ remove_blurry_frames.py
 │   └─ visualize_colmap_sparse.py
 ├─ results/
 └─ README.md
```

## 2. Collecting dataset and frame extraction

We have used a short video of a well-lighted room setting as our initial testing dataset(if we get our desired result, we are planning to use a better video with higher fps) We used Iphone with 30fps to shoot the video and put it in our data>>raw>> folder to further extract frames from it.

### Frame Extraction:
We are using cv2 library to extract frames from our video. For now the extraction rate is 5 frames per second (changeable as per requirement).
```bash
python scripts/frame_extraction.py
```
After running this script the output images can be stored in our frames folder under data.

### Blurry Frame Removal (Variance of Laplacian Method):
Before sending our frames into COLMAP, we cleaned them by removing the blurry ones. We used a simple sharpness check based on the variance of the Laplacian, which basically measures how many edges or details an image has. Sharp images have strong edges, while blurry images have very weak ones. So in the script, we loop through all frames, calculate this sharpness score, and then decide which images are “too blurry” by comparing them to an adaptive threshold based on the median sharpness of the whole set. Any frame below that threshold gets moved into a separate folder. This way, we make sure COLMAP only receives good, sharp frames, which improves the feature matching and the final reconstruction quality.
```bash
python scripts/remove_blurry_frames.py
```

## 3. Feature Extraction and Feature Matching

Once we had our cleaned frames, we used COLMAP to detect SIFT features in every image.
These features act like “landmarks” that COLMAP can track across multiple frames.
We stored everything in database.db, which COLMAP uses in the next steps.
```bash
colmap feature_extractor \
    --database_path data/colmap_output/database.db \
    --image_path data/frames/test \
    --ImageReader.single_camera 1
```
After extracting features, we told COLMAP to match them between all image pairs.
This helps COLMAP figure out which frames overlap and which points correspond to each other in 3D space.
Matching took some time, but it was a key step because without this, COLMAP can’t reconstruct the scene.
```bash
colmap exhaustive_matcher --database_path data/colmap_output/database.db
```

## 4. Sparse Reconstruction (Mapper)

With features and matches ready, we ran COLMAP’s mapper.
This is where COLMAP actually estimates:
-the camera poses for each frame
-a sparse 3D point cloud of the scene

This is basically COLMAP “solving” the geometry of the scene.
The output is written into data/colmap_output/sparse/, specifically inside the folder 0/.

```bash
colmap mapper \
    --database_path data/colmap_output/database.db \
    --image_path data/frames/test \
    --output_path data/colmap_output/sparse
```
After this we will get these files as output:
```python
cameras.bin
images.bin
points3D.bin
```

## 5. Exporting the Sparse Model to PLY

COLMAP stores reconstructions in its own binary format, so to visualize them easily,
we exported the 3D points into a .ply file.
We had to give COLMAP a full output filepath, otherwise macOS throws an error.
This generated points3D.ply, which is basically our sparse point cloud.

```bash
colmap model_converter \
    --input_path data/colmap_output/sparse/0 \
    --output_path data/colmap_output/sparse/0/points3D.ply \
    --output_type PLY
```

## 6. Visualizing the Sparse Point Cloud

We loaded the exported .ply file in a small Open3D script to see what the reconstruction looked like.

```python
import open3d as o3d
import os

# path to your exported point cloud
ply_path = "../data/colmap_output/sparse/0/points3D.ply"

if not os.path.exists(ply_path):
    print(f" File not found: {ply_path}")
else:
    print(f" Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(pcd)  # prints number of points etc.
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="COLMAP Sparse Reconstruction",
        width=1000,
        height=800,
    )
```
The visualization usually shows: a tight cluster of points, sometimes shifted away from the center or sometimes not directly recognizable.
This is normal because COLMAP only reconstructs distinctive feature points and doesn’t densify the scene.

<img width="493" height="430" alt="Screenshot 2025-12-09 at 10 57 57 AM" src="https://github.com/user-attachments/assets/0926ee16-d95c-4e0c-b8df-b23df3e26eee" />

---

## 7. Image Undistortion

After finishing the sparse reconstruction, we needed to undistort the frames because Gaussian Splatting only supports PINHOLE or SIMPLE_PINHOLE intrinsics.
COLMAP’s default SfM uses more complex models (e.g., OPENCV), so we fixed this locally.

We ran the COLMAP undistorter on our machine:

```bash
colmap image_undistorter \
    --image_path data/frames/test \
    --input_path data/colmap_output/sparse/0 \
    --output_path scene_gs \
    --output_type COLMAP
```
This created a new folder scene_gs/ with:
```python
scene_gs/
│── images/          (undistorted images)
│── sparse/
│     ├── cameras.bin
│     ├── images.bin
│     └── points3D.bin
```

### Converting COLMAP Binary Files (.bin → .txt)
Gaussian Splatting does not read .bin files, so we converted everything to text format and saved it to 0 folder in sparse.
```bash
colmap model_converter \
    --input_path scene_gs/sparse \
    --output_path scene_gs/sparse/0 \
    --output_type TXT
```
Final dataset structure:
```bash
scene_gs/
│── images/
│── sparse/
│     └── 0/
│          ├── cameras.txt
│          ├── images.txt
│          └── points3D.txt
```

## 8. Training on the GPU Server (University Cluster)
Since our local machine (Mac M2) cannot train Gaussian Splatting models, we moved the final training pipeline to the university GPU server. Below is everything we did to get training running there.

### A. Connect to the server
After connecting to the university VPN, we accessed the GPU VM using VS Code Remote. The project workspace on the server is located at /workspace/.
Inside this directory, we keep:

-the official gaussian-splatting repository

-our own gaussian_project folder (dataset + outputs)

### B. Clone the official Gaussian Splatting repository
We cloned the upstream repository as we were not able to use the CLI commands for covert,train and render also because it contains:

-the training script (train.py)

-the renderer

-the COLMAP scene loader

-CUDA-based acceleration modules needed for real-time GS

```bash
cd /workspace
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
```
This will give us full-training framework.

### C. Uploaded our prepared dataset (scene_gs) to the server
We first prepared the dataset locally (undistortion + bin→txt conversion) and then uploaded the entire folder to ""/workspace/gaussian_project/scene_gs""
Final dataset structure on the server:
```bash
scene_gs/
 ├── images/               # undistorted frames
 ├── sparse/
 │     └── 0/
 │         ├── cameras.txt
 │         ├── images.txt
 │         ├── points3D.txt
 └── (other COLMAP files)
```
This is exactly the format expected by the Gaussian Splatting training pipeline.

### D. Build the CUDA extensions required for Gaussian Splatting
The training code depends on three custom CUDA modules:

-diff-gaussian-rasterization

-simple-knn

-fused-ssim

These cannot be installed via pip; they must be compiled against the server’s CUDA + PyTorch setup.

For each module:
```bash
cd gaussian-splatting/submodules/<module-name>
/home/student/venv/bin/python setup.py build_ext --inplace
```
We repeated this for all three of them.

### E. Install missing Python packages inside the server's venv
The server had already preinstalled:

a) CUDA-enabled PyTorch

b) the venv (/home/student/venv)

c) GPU drivers

So we only needed to add these missing small packages: 
```bash
/home/student/venv/bin/pip install tqdm plyfile opencv-python joblib
```

### F. Run the Gaussian Splatting training
Once the dataset and CUDA modules were ready, we launched training:
```bash
cd /workspace/gaussian-splatting
/home/student/venv/bin/python train.py \
    -s /workspace/gaussian_project/scene_gs \
    -m /workspace/gaussian_project/output_model \
    --iterations 15000
```
This:

-loads the COLMAP scene

-initializes 3D Gaussians

-optimizes them for 15000 iterations

-stores outputs under "/workspace/gaussian_project/output_model/"

Inside this folder, the renderer also creates two subfolders:
```bash
output_model/train/
output_model/test/
```
Each subfolder contains a directory named ours_15000, which holds the rendered frames generated from the trained model.

## 9. Rendering Final Images Using the Trained Model
After training finished, we used render.py to generate the 2D renderings from the optimized 3D Gaussians.
To render the training split:
```bash
/home/student/venv/bin/python render.py \
    -m /workspace/gaussian_project/output_model \
    -s /workspace/gaussian_project/scene_gs \
    --train
```
This produced the final RGB images at:
```swift
/workspace/gaussian_project/output_model/train/ours_15000/renders/
```
These are the frames that we later assembled into a video.

## 10. Converting Rendered Frames into a Video
The frames produced by Gaussian Splatting are simply PNG images.
We used FFmpeg to turn them into an MP4 video.

Important: some images had odd dimensions, which FFmpeg does not accept for H.264 encoding, so we padded the height by 1 pixel.

From inside the directory containing the PNG files:
```bash
ffmpeg -framerate 30 -pattern_type glob -i "*.png" \
    -vf "pad=width=iw:height=ih+1" \
    -c:v libx264 -pix_fmt yuv420p output_15000.mp4
```
This produced our final rendered video **output_15000.mp4**. The generated PNG frames (and the resulting MP4 video) are not just the original video.
Instead, they represent:

-A re-rendering of the scene

-Using only the learned 3D Gaussians

-From the same camera poses that COLMAP estimated

-After 15,000 iterations of optimization

This verifies that:

-The pipeline successfully reconstructed the scene in 3D

-The Gaussian model can render synthetic views that match the input images

-The system achieves the Minimum Goal and part of the Desired Goal
