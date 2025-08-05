# 读取PATH/rand_{}_skin.obj文件中的顶点数据,存为ply文件
import os
import numpy as np

def read_obj_vertices(file_path):
    """
    读取OBJ文件中的顶点数据
    参数:
        file_path (str): OBJ文件路径
    返回:
        np.ndarray: 顶点数据，形状为(N, 3)，N为顶点数量
    """
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
    return np.array(vertices, dtype=np.float32)
def save_vertices_to_ply(vertices, output_file):
    """
    将顶点数据保存为PLY文件
    参数:
        vertices (np.ndarray): 顶点数据，形状为(N, 3)
        output_file (str): 输出的PLY文件路径
    """
    with open(output_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
def convert_obj_to_ply(obj_file, ply_file):
    """
    将OBJ文件转换为PLY文件
    参数:
        obj_file (str): 输入的OBJ文件路径
        ply_file (str): 输出的PLY文件路径
    """
    vertices = read_obj_vertices(obj_file)
    save_vertices_to_ply(vertices, ply_file)
    print(f"已将 {obj_file} 转换为 {ply_file}")
if __name__ == "__main__":
    obj_dir = "output_range1.5"
    ply_dir = "complete_pc_range1.5"
    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)
    for obj_file in os.listdir(obj_dir):
        if obj_file.endswith("skin.obj"):
            obj_path = os.path.join(obj_dir, obj_file)
            ply_file = os.path.join(ply_dir, obj_file.replace("skin.obj", "skin.ply"))
            convert_obj_to_ply(obj_path, ply_file)
            print(f"转换完成: {obj_path} -> {ply_file}")
    print("所有OBJ文件已转换为PLY文件。")
