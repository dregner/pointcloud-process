import os
import numpy as np
from src.pointcloud_viewer import PointCloudProcess
import open3d as o3d

def main():

    pcl_process = PointCloudProcess()

    passive_data = pcl_process.load_point_cloud('./data/passive_data.csv', numpy=True)
    active_data = pcl_process.load_point_cloud('./data/active_data.csv', numpy=True)
    green_colors = np.tile([0, 255, 0], (active_data.shape[0], 1))
    active_data = np.hstack((active_data[:, :3], green_colors))
    pcl_process.show_point_cloud(pcl_process.from_numpy(active_data))
    pcl_process.show_point_cloud(pcl_process.from_numpy(passive_data))
    
    T_a_p = np.array([
        [9.82936251e-01, -1.77189800e-05, -1.83946533e-01, 8.48289272e-02],
        [1.77189800e-05, 1.00000000e+00, -1.64369628e-06, -6.19984714e-02],
        [1.83946533e-01, -1.64369628e-06, 9.82936251e-01, 1.58723868e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    T_p_a = np.array([
        [9.82936251e-01, 1.77189800e-05, 1.83946533e-01, -8.62999996e-02],
        [-1.77189800e-05, 1.00000000e+00, -1.64369628e-06, 6.20000005e-02],
        [-1.83946533e-01, -1.64369628e-06, 9.82936251e-01, 2.34074107e-06],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    pcl_active2passive = pcl_process.transform_point_cloud(active_data, T_p_a)
    # pcl_process.show_point_cloud(pcl_process.from_numpy(pcl_active2passive))    
    concat = np.vstack((pcl_active2passive, passive_data))
    pcl_process.show_point_cloud(pcl_process.from_numpy(concat))
    new_active = pcl_process.interative_closest_point(pcl_active2passive, passive_data, threshold=0.05)
    new_concat = np.vstack((new_active, passive_data))
    pcl_process.show_point_cloud(pcl_process.from_numpy(new_concat))

    # Considere apenas as colunas X e Y
    xy_active = np.round(new_active[:, :2], decimals=2)
    xy_passive = np.round(passive_data[:, :2], decimals=2)

    # Converta para tuplas para facilitar a comparação
    xy_active_tuples = set(map(tuple, xy_active))
    xy_passive_tuples = set(map(tuple, xy_passive))

    # Encontre os pontos XY comuns
    common_xy = np.array(list(xy_active_tuples & xy_passive_tuples))
    print(f"Número de pontos XY comuns: {common_xy.shape[0]}")
    passive_colors = passive_data[:, 3:].copy()
    for xy in common_xy:
        idx = np.where(
            (np.round(passive_data[:, 0], 1) == xy[0]) &
            (np.round(passive_data[:, 1], 1) == xy[1])
        )[0]
        passive_colors[idx] = [255, 0, 0]  # vermelho

    passive_colored = np.hstack((passive_data[:, :3], passive_colors))
    final_concat = np.vstack((new_active, passive_colored))
    pcl_process.show_point_cloud(pcl_process.from_numpy(final_concat))

    
if __name__ == "__main__":
    main()
