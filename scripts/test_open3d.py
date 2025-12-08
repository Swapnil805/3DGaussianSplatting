import open3d as o3d
import numpy as np

# Create a sample 3D point cloud
points = np.random.rand(1000, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
