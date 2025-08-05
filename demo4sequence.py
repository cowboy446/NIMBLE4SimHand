import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes


if __name__ == "__main__":
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
    
    # total_samples = 100  # 总共要生成的样本数
    batch_size = 100  # 每次处理的批次大小
    sequence_num = 100  # 序列的数量
    total_samples = sequence_num * batch_size  # 总样本数
    delta = 0.1
    output_folder = "output_range{}_seq_new".format(delta)
    os.makedirs(output_folder, exist_ok=True)
    
    # 使用循环分批处理，减少内存占用
    for batch_idx in range(0, total_samples, batch_size):
        seq_idx = batch_idx // batch_size + 1
        print(f"处理批次 {batch_idx//batch_size + 1}/{total_samples//batch_size}...")
        bn = min(batch_size, total_samples - batch_idx)  # 确保最后一批正确

        pose_param = torch.zeros(bn, 30, device=device)
        # pose_param[0] = torch.rand(30, device=device) * 2 * 1 - 1  # 第一个样本随机生成

        for i in range(1, bn):
            # 生成随机变化，确保正负平衡
            change = torch.randn(30, device=device) * 2 * delta - delta
            positive_mask = torch.rand(30, device=device) > 0.5
            change[positive_mask] = torch.abs(change[positive_mask])  # 保证部分变化为正数
            change[~positive_mask] = -torch.abs(change[~positive_mask])  # 保证部分变化为负数
            
            # 更新 pose_param
            pose_param[i] = pose_param[i-1] + change
            # 限制范围
            pose_param[i] = torch.clamp(pose_param[i], -1, 1)
        # for i in range(bn):
        #     pose_param[i, i] = 1  # 逐渐增加第一个参数的值
        shape_param = torch.zeros(bn, 20, device=device)
        tex_param = torch.zeros(bn, 10, device=device)
        
        # 生成模型
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
        
        # 创建网格
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(bn, 1, 1))
        muscle_p3dmesh = Meshes(muscle_v, nlayer.muscle_f.repeat(bn, 1, 1))
        bone_p3dmesh = Meshes(bone_v, nlayer.bone_f.repeat(bn, 1, 1))
        
        # 平滑网格
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        muscle_p3dmesh = smooth_mesh(muscle_p3dmesh)
        bone_p3dmesh = smooth_mesh(bone_p3dmesh)
        
        # 转换到MANO顶点
        skin_mano_v = nlayer.nimble_to_mano(skin_v, is_surface=True)
        
        # 准备保存数据
        tex_img = tex_img.detach().cpu().numpy()
        skin_v_smooth = skin_p3dmesh.verts_padded().detach().cpu().numpy()
        bone_joints = bone_joints.detach().cpu().numpy()
        
        # 保存生成的模型数据
        for i in range(bn):
            sample_idx = i
            # 创建子文件夹
            joints25_dir = os.path.join(output_folder, "joints25")
            joints21_dir = os.path.join(output_folder, "joints21")
            manov_dir = os.path.join(output_folder, "manov")
            bone_dir = os.path.join(output_folder, "bone")
            muscle_dir = os.path.join(output_folder, "muscle")
            skin_obj_dir = os.path.join(output_folder, "skin")
            param_dir = os.path.join(output_folder, "param")
            os.makedirs(param_dir, exist_ok=True)
            os.makedirs(joints25_dir, exist_ok=True)
            os.makedirs(joints21_dir, exist_ok=True)
            os.makedirs(skin_obj_dir, exist_ok=True)
            # os.makedirs(manov_dir, exist_ok=True)
            # os.makedirs(bone_dir, exist_ok=True)
            # os.makedirs(muscle_dir, exist_ok=True)

            np.savetxt(os.path.join(joints25_dir, f"seq{seq_idx}_frame{sample_idx}_joints25.xyz"), bone_joints[i])
            print(f"Saved joints25 to {os.path.join(joints25_dir, f'seq{seq_idx}_frame{sample_idx}_joints25.xyz')}")
            # 对于joints21,在joints25的基础上去掉第5, 10, 15, 20个关节(以0为起点)
            joints21 = np.delete(bone_joints[i], [5, 10, 15, 20], axis=0)
            np.savetxt(os.path.join(joints21_dir, f"seq{seq_idx}_frame{sample_idx}_joints21.xyz"), joints21)
            # np.savetxt(os.path.join(manov_dir, f"seq{seq_idx}_frame{sample_idx}_manov.xyz"), skin_mano_v[i].detach().cpu().numpy())  # 添加 .detach().cpu().numpy()
            # pytorch3d.io.IO().save_mesh(bone_p3dmesh[i], os.path.join(bone_dir, f"seq{seq_idx}_frame{sample_idx}_bone.obj"))
            # pytorch3d.io.IO().save_mesh(muscle_p3dmesh[i], os.path.join(muscle_dir, f"seq{seq_idx}_frame{sample_idx}_muscle.obj"))
            save_textured_nimble(os.path.join(skin_obj_dir, f"seq{seq_idx}_frame{sample_idx}.obj"), skin_v_smooth[i], tex_img[i])
            # 保存参数
            np.savetxt(os.path.join(param_dir, f"seq{seq_idx}_frame{sample_idx}_pose.txt"), pose_param[i].detach().cpu().numpy())
            np.savetxt(os.path.join(param_dir, f"seq{seq_idx}_frame{sample_idx}_shape.txt"), shape_param[i].detach().cpu().numpy())
            np.savetxt(os.path.join(param_dir, f"seq{seq_idx}_frame{sample_idx}_tex.txt"), tex_param[i].detach().cpu().numpy())
            print(f"Saved parameters for seq{seq_idx}, frame {sample_idx} to {param_dir}")
        # 清理GPU内存
        torch.cuda.empty_cache()
        print(f"批次 {batch_idx//batch_size + 1} 完成，已保存 {min(batch_idx + bn, total_samples)} 个样本")