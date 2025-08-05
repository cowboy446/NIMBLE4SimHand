import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_hand_mesh(vertices, faces, title):
    """Visualize a hand mesh"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh as a collection of triangles
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # Normalize the vertices to [-1, 1] for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Plot the triangular mesh
    tri = ax.plot_trisurf(x, y, z, triangles=faces, 
                         cmap=plt.cm.Spectral, alpha=0.7)
    
    # Add a colorbar for reference
    fig.colorbar(tri, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Save the figure
    plt.savefig(f"pose_visualization/{title}.png", bbox_inches='tight')
    plt.close()

def explore_pca_components():
    """Explore each PCA component individually"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pm_dict_name = r"assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"assets/NIMBLE_TEX_DICT.pkl"

    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

    if os.path.exists(r"assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    else:
        nimble_mano_vreg=None

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    
    # 创建输出文件夹
    os.makedirs("pose_visualization", exist_ok=True)
    
    # 零姿态作为基准
    shape_param = torch.zeros(1, 20, device=device)  # 使用平均手形
    tex_param = torch.zeros(1, 10, device=device)    # 使用平均纹理
    
    # 生成零姿态（参考姿态）
    pose_param = torch.zeros(1, 30, device=device)
    skin_v, _, _, _, _ = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
    skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
    skin_p3dmesh = smooth_mesh(skin_p3dmesh)
    skin_v_ref = skin_p3dmesh.verts_padded()[0].detach().cpu().numpy()
    skin_f_ref = nlayer.skin_f[0].detach().cpu().numpy()
    
    # 可视化参考姿态
    visualize_hand_mesh(skin_v_ref, skin_f_ref, "reference_pose")
    
    # 探索每个主成分
    for i in range(30):
        for value in [-3.0, -1.5, 1.5, 3.0]:
            # 创建只有一个主成分激活的姿态参数
            pose_param = torch.zeros(1, 30, device=device)
            pose_param[0, i] = value
            
            # 生成手部姿态
            skin_v, _, _, _, _ = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
            
            # 创建和平滑网格
            skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
            skin_p3dmesh = smooth_mesh(skin_p3dmesh)
            skin_v_i = skin_p3dmesh.verts_padded()[0].detach().cpu().numpy()
            skin_f_i = nlayer.skin_f[0].detach().cpu().numpy()
            
            # 可视化
            visualize_hand_mesh(skin_v_i, skin_f_i, f"pca_{i}_value_{value}")
            
            # 也保存为OBJ文件以便更仔细检查
            pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], os.path.join("pose_visualization", f"pca_{i}_value_{value}.obj"))
            
            print(f"Processed PCA component {i}, value {value}")
    
    print("PCA component exploration completed!")

def generate_number_gestures():
    """Generate hand gestures for numbers 1-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pm_dict_name = r"assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"assets/NIMBLE_TEX_DICT.pkl"

    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

    if os.path.exists(r"assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    else:
        nimble_mano_vreg=None

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    
    # 创建输出文件夹
    os.makedirs("number_gestures", exist_ok=True)
    
    # 使用平均手形和纹理
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    # 根据PCA探索的结果，我们可以为数字1-10定义姿态
    # 这些值需要根据explore_pca_components函数的输出来调整
    number_poses = {
        # 这些值仅为示例，需要根据PCA探索结果调整
        1: {0: 2.0, 5: -2.0, 7: -2.0, 12: -2.0, 15: -2.0},  # 食指伸出，其他指头弯曲
        2: {0: 2.0, 5: 2.0, 7: -2.0, 12: -2.0, 15: -2.0},   # 食指和中指伸出
        3: {0: 2.0, 5: 2.0, 7: 2.0, 12: -2.0, 15: -2.0},    # 食指、中指和无名指伸出
        4: {0: 2.0, 5: 2.0, 7: 2.0, 12: 2.0, 15: -2.0},     # 除小拇指外都伸出
        5: {0: 2.0, 5: 2.0, 7: 2.0, 12: 2.0, 15: 2.0},      # 所有手指伸出
        # 这些值需要根据实际PCA探索结果来调整
        6: {0: -2.0, 2: 2.0, 5: 2.0, 7: 2.0, 12: 2.0, 15: 2.0},
        7: {0: -2.0, 2: -2.0, 5: 2.0, 7: 2.0, 12: 2.0, 15: 2.0},
        8: {0: -2.0, 2: -2.0, 5: -2.0, 7: 2.0, 12: 2.0, 15: 2.0},
        9: {0: -2.0, 2: -2.0, 5: -2.0, 7: -2.0, 12: 2.0, 15: 2.0},
        10: {0: -2.0, 2: -2.0, 5: -2.0, 7: -2.0, 12: -2.0, 15: 2.0}
    }
    
    for number, pose_dict in number_poses.items():
        # 创建姿态参数
        pose_param = torch.zeros(1, 30, device=device)
        
        # 设置特定的PCA值
        for component, value in pose_dict.items():
            pose_param[0, component] = value
        
        # 生成手部姿态
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
        
        # 创建和平滑网格
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        # 保存OBJ文件
        output_file = os.path.join("number_gestures", f"number_{number}.obj")
        pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
        
        # 可视化
        skin_v_num = skin_p3dmesh.verts_padded()[0].detach().cpu().numpy()
        skin_f_num = nlayer.skin_f[0].detach().cpu().numpy()
        visualize_hand_mesh(skin_v_num, skin_f_num, f"number_{number}")
        
        print(f"Generated hand gesture for number {number}")
    
    print("Number gesture generation completed!")

if __name__ == "__main__":
    print("Starting PCA component exploration...")
    explore_pca_components()
    
    # print("\nStarting number gesture generation...")
    # generate_number_gestures()
