import os
import numpy as np
from src.pointcloud_process import PointCloudProcess
import open3d as o3d
from scipy.spatial  import cKDTree
def main():

    pcl_process = PointCloudProcess()
    for i in range(1,5):

        passive_data = pcl_process.load_point_cloud('./data/passive_{}.csv'.format(i), numpy=True)
        active_data = pcl_process.load_point_cloud('./data/active_{}.csv'.format(i), numpy=True)
        green_colors = np.tile([0, 255, 0], (active_data.shape[0], 1))
        active_data = np.hstack((active_data[:, :3], green_colors))
        
        T_p_a = np.array([[ 9.54892134e-01,  2.86045190e-05,  2.96952877e-01, -1.00007706e-01],
                        [-2.86045190e-05,  1.00000000e+00, -4.34509611e-06,  6.39922945e-02],
                        [-2.96952877e-01, -4.34509611e-06,  9.54892135e-01, -7.99965315e-02],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        pcl_active2passive = pcl_process.transform_point_cloud(active_data, T_p_a)
        # pcl_process.show_point_cloud(pcl_process.from_numpy(pcl_active2passive))    

        new_active = pcl_process.interative_closest_point(pcl_active2passive, passive_data, threshold=0.05)

        passive_data_kdtree, passive_data_kdtree_excluded, mask = pcl_process.remove_overlap_by_kdtree(new_active, passive_data, radius=0.02)
        pcl_process.show_point_cloud(pcl_process.from_numpy(passive_data_kdtree_excluded))

        colored_passive_data_excluded = passive_data_kdtree_excluded.copy()
        colored_passive_data_excluded[:, 3:] = [255, 0, 0]  # vermelho

        concat_kdtree = np.vstack((new_active, passive_data_kdtree, colored_passive_data_excluded))
        pcl_process.show_point_cloud(pcl_process.from_numpy(concat_kdtree))

        pcl_process.transfer_colors_to_active_interpolated(new_active, passive_data_kdtree_excluded)
        pcl_process.show_point_cloud(pcl_process.from_numpy(new_active))
        final_concat = np.vstack((new_active, passive_data_kdtree))
        pcl_process.show_point_cloud(pcl_process.from_numpy(final_concat))
        
if __name__ == "__main__":
    main()
