import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
from utils import *
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

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=False, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)

    seq_num = 1  # 序列数量
    batch_size = 3  # 每次处理的批次大小
    total_samples = seq_num * batch_size  # 总样本数
    output_folder = "output_handcraft"
    os.makedirs(output_folder, exist_ok=True)
    # 使用循环分批处理，减少内存占用
    for batch_idx in range(0, seq_num):
        print(f"处理批次 {batch_idx//batch_size + 1}/{total_samples//batch_size}...")
        bn = min(batch_size, total_samples - batch_idx)  # 确保最后一批正确

        # 模板：用方向和模长控制所有关节旋转
        # 方向 shape: (bn, 20, 3), 长度 shape: (bn, 20)
        direction = np.zeros((bn, 20, 3), dtype=np.float32)
        length = np.ones((bn, 20), dtype=np.float32)
        # 按图中参数填充
        for bidx in range(bn):
            direction[bidx, 0] = [0.00, 0.00, 0.00]
            direction[bidx, 1] = [0.00, 0.00, 0.00]
            direction[bidx, 2] = [0.00, -1.00, 0.00]
            direction[bidx, 3] = [-1.00, 0.00, 0.00]
            direction[bidx, 4] = [0.00, 0.00, 0.00]
            direction[bidx, 5] = [0.00, 0.00, 0.00]
            direction[bidx, 6] = [0.00, 0.00, 0.00]
            direction[bidx, 7] = [0.00, 0.00, 0.00]
            direction[bidx, 8] = [0.00, 0.00, 0.00]
            direction[bidx, 9] = [0.00, 0.00, 0.00]
            direction[bidx,10] = [0.00, 0.00, 0.00]
            direction[bidx,11] = [0.00, 0.00, 0.00]
            direction[bidx,12] = [0.00, 0.00, 0.00]
            direction[bidx,13] = [1.00, 0.00, 0.00]
            direction[bidx,14] = [1.00, 0.00, 1.00]
            direction[bidx,15] = [1.00, 0.00, 0.00]
            direction[bidx,16] = [0.00, 0.00, 0.00]
            direction[bidx,17] = [1.00, 0.00, 0.00]
            direction[bidx,18] = [1.00, 0.00, 1.00]
            direction[bidx,19] = [1.00, 0.00, 1.00]
            # 长度全部为1
            length[bidx, :] = bidx / 2.0
            print("bidx: ", bidx, "length: ", length[bidx, 0])
        # 归一化方向后乘以模长，得到轴角
        # 将方向和长度从tensor转换为numpy数组

        norm = np.linalg.norm(direction, axis=2, keepdims=True)
        norm[norm < 1e-8] = 1.0  # 防止除零
        axis = direction / norm
        axis_angle = axis * length[..., np.newaxis]  # (bn, 20, 3)
        pose_param = torch.tensor(axis_angle, dtype=torch.float32, device=device)

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
            seq_idx = 0  # 如果有多序列可自行调整
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
            # np.savetxt(os.path.join(manov_dir, f"seq{seq_idx}_frame{sample_idx}_manov.xyz"), skin_mano_v[i].detach().cpu().numpy())
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







