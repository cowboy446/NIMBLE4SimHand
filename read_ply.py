# 读取ply中的点云为numpy数组
import numpy as np
import open3d as o3d
def read_ply(file_path):
    """
    读取PLY文件中的点云数据
    参数:
        file_path (str): PLY文件路径
    返回:
        np.ndarray: 点云数据，形状为(N, 3)，N为点的数量
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points, dtype=np.float32)
    except Exception as e:
        print(f"读取PLY文件时发生错误: {e}")
        return None
if __name__ == "__main__":
    # 示例用法
    ply_file = "complete_pc/rand_0_skin.ply"  # 替换为你的PLY文件路径
    points = read_ply(ply_file)
    if points is not None:
        print(f"读取到 {len(points)} 个点云数据点。")
        print(points[:5])  # 打印前5个点
    else:
        print("未能读取点云数据。")
# 该函数读取PLY文件中的点云数据，并返回一个numpy数组，数组的形状为(N, 3)，其中N是点的数量。函数使用Open3D