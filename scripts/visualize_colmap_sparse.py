import open3d as o3d
import os

# path to your exported point cloud
ply_path = "../data/colmap_output/sparse/0/points3D.ply"

if not os.path.exists(ply_path):
    print(f"❌ File not found: {ply_path}")
else:
    print(f"✅ Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(pcd)  # prints number of points etc.
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="COLMAP Sparse Reconstruction",
        width=1000,
        height=800,
    )
