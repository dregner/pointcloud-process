import numpy as np
import open3d as o3d
import os


class PointCloudProcess:
    def __init__(self):
        pass

    def load_point_cloud(self, file_path, numpy=False):
        """
        Load a point cloud from a file.
        Supported formats: .ply, .pcd, .xyz, .xyzrgb, .xyzn, .pts, and .csv ([x y z] or [x y z r g b]).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            if numpy:
                return data
            else:
                return self.from_numpy(data)
        else:
            pcd = o3d.io.read_point_cloud(file_path)
            return pcd

    def show_point_cloud(self, pcd):
        """
        Visualize a point cloud using Open3D.
        """
        if isinstance(pcd, o3d.geometry.PointCloud):
            o3d.visualization.draw_geometries([pcd])
        else:
            raise TypeError("Input must be an Open3D PointCloud object.")

    def from_numpy(self, points: np.ndarray):
        """
        Create an Open3D point cloud from a numpy array of shape (N, 3) or (N, 6).
        """
        if points.ndim != 2 or points.shape[1] not in [3, 6]:
            raise ValueError("Input numpy array must have shape (N, 3) or (N, 6).")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] == 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
        return pcd

    def transform_point_cloud(self, pointcloud, transform):
        """
        Appy transformation matrix to numpy pointcloud
        Input:
            pointcloud: (N, 3) or (N, 6) numpy array
            transform: (4, 4) transformation matrix
        Output:
            transformed pointcloud: (N, 3) or (N, 6) numpy array
        """
        xyz = pointcloud[:, :3]
        ones = np.ones((xyz.shape[0], 1))
        xyz1 = np.hstack((xyz, ones))
        transformed_xyz = (transform @ xyz1.T).T
        pointcloud[:, :3] = transformed_xyz[:, :3]

        return pointcloud
    
    def concatenate_pointclouds(self):
        self.get_logger().info('Concatenating point clouds...')

        pcd_sm4 = self.interative_closest_point(self.transformed_active_xyz, self.transformed_passive_xyz, threshold=0.05)

        # Concatenate point clouds
        aligned_sm4 = np.asarray(pcd_sm4.points, dtype=np.float32)

        combined_xyz = np.vstack((self.transformed_passive_xyz, aligned_sm4))

        # Combine RGB data
        rgb_sm4 = np.zeros(aligned_sm4.shape[0], dtype=np.float32) 
        combined_rgb = np.hstack((self.points_rgb, rgb_sm4))


        concatenated_pcd = np.hstack((combined_xyz, combined_rgb.reshape(-1, 1)))
        return concatenated_pcd
    
    def interative_closest_point(self, source, target, threshold=0.05):
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source[:,:3])

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target[:,:3])

        reg_icp = o3d.pipelines.registration.registration_icp(  
            pcd_source, pcd_target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcd_source.transform(reg_icp.transformation)
        source_icp = np.hstack((np.asarray(pcd_source.points), source[:, 3:])) if source.shape[1] > 3 else np.asarray(pcd_source.points)

        return source_icp
    
    def save_point_cloud(self, pcd, file_path):
        """
        Save a point cloud to a file.
        """
        if not isinstance(pcd, o3d.geometry.PointCloud):
            pcd = self.from_numpy(pcd)
        o3d.io.write_point_cloud(file_path, pcd)