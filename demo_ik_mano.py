import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer
# import sys
# import os
# sys.path.append(os.path.abspath("."))
from mano.our_mano import OurManoLayer

# from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
from utils_for_mano import *
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes
import open3d as o3d


# def batch_rodrigues_simple(axisang):
#     """Convert axis-angle to rotation matrix"""
#     batch_size = axisang.shape[0]
#     angle = torch.norm(axisang + 1e-8, dim=1, keepdim=True)
#     normalized = torch.div(axisang, angle)
#     angle = angle * 0.5
#     v_cos = torch.cos(angle)
#     v_sin = torch.sin(angle)
#     quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    
#     # Convert quaternion to rotation matrix
#     w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
#     w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
#     wx, wy, wz = w * x, w * y, w * z
#     xy, xz, yz = x * y, x * z, y * z

#     rotMat = torch.stack([
#         w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
#         2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
#         2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
#     ], dim=1).view(batch_size, 3, 3)
    
#     return rotMat


# import torch

import torch

def batch_rodrigues(axis_angle: torch.Tensor, eps: float = 1e-8):
    """
    axis_angle: [B,3] -> rotation matrices [B,3,3]
    A simple, vectorized Rodrigues formula with small-angle epsilon.
    """
    assert axis_angle.dim() == 2 and axis_angle.size(1) == 3
    B = axis_angle.shape[0]
    theta = torch.norm(axis_angle, dim=1, keepdim=True)  # [B,1]
    k = axis_angle / (theta + eps)  # [B,3]

    x = k[:, 0].view(B, 1, 1)
    y = k[:, 1].view(B, 1, 1)
    z = k[:, 2].view(B, 1, 1)

    # skew-symmetric K
    K = torch.zeros(B, 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[:, 0, 1] = -z.view(-1)
    K[:, 0, 2] = y.view(-1)
    K[:, 1, 0] = z.view(-1)
    K[:, 1, 2] = -x.view(-1)
    K[:, 2, 0] = -y.view(-1)
    K[:, 2, 1] = x.view(-1)

    theta_expand = theta.view(B, 1, 1)  # [B,1,1]
    sin_t = torch.sin(theta_expand)
    cos_t = torch.cos(theta_expand)
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).repeat(B, 1, 1)

    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R  # [B,3,3]


def simple_forward_kinematics(pose_param_16x3: torch.Tensor, rest_joints: torch.Tensor, device=None):
    """
    改动说明：
      - 把 16 维按 (root, per-finger: MCP, PIP, DIP) 映射到 joint indices:
          root -> 0
          thumb: 1,2,3
          index: 5,6,7
          middle:9,10,11
          ring:13,14,15
          little:17,18,19
      - 这样 pose[:,1,:] 会控制 joint 1（MCP），满足你的预期。
      - 位置更新使用 parent 的 global rotation： pos[j] = pos[parent] + parent_rot @ (rest[j]-rest[parent])
      - 全局旋转传播： global_rot[j] = parent_rot @ local_rot[j]
    Args:
      pose_param_16x3: [B,16,3]
      rest_joints: [B,21,3] （绝对 rest 坐标，函数会用差值得到相对偏移）
    Returns:
      joint_positions: [B,21,3]
    """
    if device is None:
        device = pose_param_16x3.device
    B = pose_param_16x3.shape[0]

    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],      # 食指
        [0, 9], [9, 10], [10, 11], [11, 12], # 中指
        [0, 13], [13, 14], [14, 15], [15, 16], # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
    ]
    parent_idx = [-1] * 21
    for p, c in bones:
        parent_idx[c] = p

    # —— 关键映射（已改为 MCP 可动） —— 
    joint_to_param_idx = {
        0: 0,   # global root

        # 拇指（MCP, PIP, DIP）
        1: 1, 2: 2, 3: 3,
        # 食指
        5: 4, 6: 5, 7: 6,
        # 中指
        9: 7, 10: 8, 11: 9,
        # 无名指
        13: 10, 14: 11, 15: 12,
        # 小指
        17: 13, 18: 14, 19: 15,
        # 关节 4,8,12,16,20 (tips) 保持不映射（无参数）
    }

    joint_positions = torch.zeros(B, 21, 3, device=device, dtype=rest_joints.dtype)
    transforms = [None] * 21  # 每项存 Bx3x3 的 global rotation

    # 根关节 local rotation（若没有参数则为 I）
    if 0 in joint_to_param_idx:
        root_local = pose_param_16x3[:, joint_to_param_idx[0], :]  # [B,3]
        root_rot = batch_rodrigues(root_local)  # [B,3,3]
    else:
        root_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

    # 根位置：旋转 rest_joints[0]（若你希望根不旋转可改为直接赋值）
    joint_positions[:, 0, :] = torch.bmm(root_rot, rest_joints[:, 0, :].unsqueeze(-1)).squeeze(-1)
    transforms[0] = root_rot

    # 逐关节递归
    for j in range(1, 21):
        parent = parent_idx[j]
        parent_pos = joint_positions[:, parent, :]       # [B,3]
        parent_rot = transforms[parent]                  # [B,3,3]

        rel = rest_joints[:, j, :] - rest_joints[:, parent, :]  # [B,3]

        if j in joint_to_param_idx:
            local_axis_angle = pose_param_16x3[:, joint_to_param_idx[j], :]  # [B,3]
            local_rot = batch_rodrigues(local_axis_angle)  # [B,3,3]
        else:
            local_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

        # global rotation for this joint (for its children)
        global_rot = torch.bmm(parent_rot, local_rot)  # [B,3,3]
        transforms[j] = global_rot

        # joint position only affected by parent's global rotation
        rotated_rel = torch.bmm(parent_rot, rel.unsqueeze(-1)).squeeze(-1)  # [B,3]
        joint_positions[:, j, :] = parent_pos + rotated_rel

    return joint_positions


# # small helper to accept batch_rodrigues usage above
# def batch_rodrigues(axis_angle: torch.Tensor, eps: float = 1e-8):
#     """
#     支持 axis_angle: [B,3]
#     如果你想对 [B,N,3] 一次性求解，可用 loop 或把维度合并后 reshape 回去。
#     """
#     return batch_rodrigues.__wrapped__(axis_angle, eps) if hasattr(batch_rodrigues, '__wrapped__') else batch_rodrigues(axis_angle, eps)


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
    nimble_pose_aa = torch.zeros(1, 20, 3, device=device)  # 初始化姿态参数
    nimble_shape_param = torch.zeros(1, 20, device=device)  #
    _, rest_joints = nlayer.generate_hand_shape(nimble_shape_param, normalized=True)
    rest_joints = rest_joints[:,[0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24], :] 
    # mm = OurManoLayer().to(device)

    

    total_samples = 1  # 总共要生成的样本数
    batch_size = 1  # 每次处理的批次大小
    output_folder = "output_ik_mano"
    os.makedirs(output_folder, exist_ok=True)
    pose_param = torch.zeros(1, 16, 3, device=device)

    # pose_param[:,0,:] = torch.tensor([2, 0, 0], device=device)  # 设置第2个关节的旋转
    pose_param[:,4,:] = torch.tensor([2, 0, 0], device=device)  # 设置第3个关节的旋转
    # pose_param[:,2,:] = torch.tensor([2, 0, 0], device=device)  # 设置第4个关节的旋转
    shape_param = torch.zeros(1, 10, device=device)
    # 将pose_param转为Bx48
    pose_param_full = torch.zeros(1, 48, device=device)
    pose_param_full[:, :16*3] = pose_param.reshape(1, -1)
    th_trans = torch.zeros(1, 3, device=device)
    # _, rest_joints = mm.forward(pose_param_full, True, shape_param, th_trans)
    # print(f"rest_joints: {rest_joints}")
    
    # 使用simplified forward kinematics计算当前pose的关节位置
    current_joints = simple_forward_kinematics(pose_param, rest_joints, device)
    print(f"current_joints shape: {current_joints.shape}")
    
    # o3d visualization
    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]

    # 可视化当前pose
    current_joints_np = current_joints.cpu().numpy()
    print(f"current_joints_np shape: {current_joints_np.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(current_joints_np[0])
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(current_joints_np[0])
    lines.lines = o3d.utility.Vector2iVector(bones)
    lines.paint_uniform_color([1, 0, 0])  # 红色骨骼 - 当前pose
    o3d.visualization.draw_geometries([pcd, lines])
    kpt_3d = np.loadtxt("output_range0.1_seq5/joints25/seq10_frame1_joints25.xyz")
    kpt_3d = torch.tensor(kpt_3d, dtype=torch.float32, device=device)
    kpt_3d = kpt_3d.unsqueeze(0).repeat(1, 1, 1)  # 扩展到批次大小
    print(f"kpt_3d shape: {kpt_3d.shape}")
    # 去掉第5,10,15,20个关节点
    kpt_mano = kpt_3d[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24], :]
    print(f"kpt_mano shape: {kpt_mano.shape}")
    rest_mano  = rest_joints
    print(f"rest_mano shape: {rest_mano.shape}")
    mano_aa,_,_ = posevec_2axisang_mano(kpt_mano, rest_mano)
    # 将mano_aa reshape为16x3格式
    mano_aa_16x3 = mano_aa.view(1, 16, 3)
    print(f"mano_aa_16x3 shape: {mano_aa_16x3.shape}")
    
    # 使用simplified forward kinematics计算IK结果
    joints_ik_simple = simple_forward_kinematics(mano_aa_16x3, rest_joints, device)
    print(f"joints_ik_simple shape: {joints_ik_simple.shape}")
    
    # 可视化IK结果
    joints_ik_simple_np = joints_ik_simple.cpu().numpy()
    print(f"joints_ik_simple_np shape: {joints_ik_simple_np.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joints_ik_simple_np[0])
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(joints_ik_simple_np[0])
    lines.lines = o3d.utility.Vector2iVector(bones)
    lines.paint_uniform_color([0, 1, 0])  # 绿色骨骼 - IK结果
    o3d.visualization.draw_geometries([pcd, lines]) 
    