'''
    NIMBLE: A Non-rigid Hand Model with Bones and Muscles[SIGGRAPH-22]
    https://reyuwei.github.io/proj/nimble
'''

import torch
import pytorch3d
import numpy as np
from pathlib import Path
from pytorch3d.structures.meshes import Meshes
import pytorch3d.ops
import os

ROOT_JOINT_IDX = 0  # wrist
DOF2_BONES = [1, 2, 4, 5, 8, 9, 12, 13, 16, 17]
DOF1_BONES = [3, 6, 7, 10, 11, 14, 15, 18, 19]
JOINT_PARENT_ID_DICT = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,

    5: 0,
    6: 5,
    7: 6,
    8: 7,
    9: 8,

    10: 0,
    11: 10,
    12: 11,
    13: 12,
    14: 13,

    15: 0,
    16: 15,
    17: 16,
    18: 17,
    19: 18,

    20: 0,
    21: 20,
    22: 21,
    23: 22,
    24: 23
}
JOINT_ID_NAME_DICT = {
    0: "carpal",
    1: "met1",
    2: "pro1",
    3: "dis1",
    4: "dis1_end",

    5: "met2",
    6: "pro2",
    7: "int2",
    8: "dis2",
    9: "dis2_end",

    10: "met3",
    11: "pro3",
    12: "int3",
    13: "dis3",
    14: "dis3_end",

    15: "met4",
    16: "pro4",
    17: "int4",
    18: "dis4",
    19: "dis4_end",

    20: "met5",
    21: "pro5",
    22: "int5",
    23: "dis5",
    24: "dis5_end"
}
BONE_TO_JOINT_NAME = {
    0: "carpal",

    1: "met1",
    2: "pro1",
    3: "dis1",

    4: "met2",
    5: "pro2",
    6: "int2",
    7: "dis2",

    8: "met3",
    9: "pro3",
    10: "int3",
    11: "dis3",

    12: "met4",
    13: "pro4",
    14: "int4",
    15: "dis4",

    16: "met5",
    17: "pro5",
    18: "int5",
    19: "dis5",
}
STATIC_BONE_NUM = 20
STATIC_JOINT_NUM = 25
JOINT_ID_BONE_DICT = {}
JOINT_ID_BONE = np.zeros(STATIC_BONE_NUM)
BONE_ID_JOINT_DICT = {}
for key in JOINT_ID_NAME_DICT:
    value = JOINT_ID_NAME_DICT[key]
    for key_b in BONE_TO_JOINT_NAME:
        if BONE_TO_JOINT_NAME[key_b] == value:
            JOINT_ID_BONE_DICT[key] = key_b
            BONE_ID_JOINT_DICT[key_b] = key
            JOINT_ID_BONE[key_b] = key

def dis_to_weight(dismat, thres_corres, node_sigma):
    dismat[dismat==0] = 1e5
    dismat[dismat>thres_corres] = 1e5
    node_weight = torch.exp(-dismat / node_sigma)
    norm = torch.norm(node_weight, dim=1)
    norm_node_weight = node_weight / (norm + 1e-6)
    norm_node_weight[norm==0] = 0
    return norm_node_weight

def batch_to_tensor_device(batch, device):
    def to_tensor(arr):
        if isinstance(arr, int):
            return arr
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        if arr.dtype == np.int64:
            arr = torch.from_numpy(arr)
        else:
            arr = torch.from_numpy(arr).float()
        return arr

    for key in batch:
        if isinstance(batch[key], np.ndarray):
            batch[key] = to_tensor(batch[key]).to(device)
        elif isinstance(batch[key], list):
            for i in range(len(batch[key])):
                if isinstance(batch[key][i], list):
                    for j in range(len(batch[key][i])):
                        if isinstance(batch[key][i][j], np.ndarray):
                            batch[key][i][j] = to_tensor(batch[key][i][j]).to(device)
                else:
                    batch[key][i] = to_tensor(batch[key][i]).to(device)
        elif isinstance(batch[key], dict):
            batch[key] = batch_to_tensor_device(batch[key], device)
        elif isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    return batch

def quat2aa(quats):
    """
    Convert wxyz quaternions to angle-axis representation
    :param quats:
    :return:
    """
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = torch.atan2(_sin, _cos) * 2
    return axis * angle.unsqueeze(-1)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_aa2quat(axisang):
    # w, x, y, z
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return quat

def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def th_posemap_axisang_2output(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb - 1):
        joint_idx_val = joint_idx + 1
        axis_ang = pose_vectors[:, joint_idx_val * 3:(joint_idx_val + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    # rot_mats = torch.stack(rot_mats, 1).view(-1, 15 *9)
    rot_mats = torch.cat(rot_mats, 1)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats

def posevec_2axisang(pose_3d, rest_pose_3d):
    exclude_children = [1, 5, 10, 15, 20]  # wrist and thumb base
    used_children = [i for i in range(25) if i not in exclude_children]
    axis_ang_list = torch.zeros((pose_3d.shape[0], len(used_children), 3), dtype=pose_3d.dtype, device=pose_3d.device)
    exclude_axis_ang_list = torch.zeros((pose_3d.shape[0], len(exclude_children), 3), dtype=pose_3d.dtype, device=pose_3d.device)
    for i, idx in enumerate(used_children):
        # print("i = ", i, "idx = ", idx)
        parent = JOINT_PARENT_ID_DICT[idx]
        if parent == -1:
            continue
        # print(pose_3d.shape, rest_pose_3d.shape)
        p_parent = pose_3d[:, parent, :]
        p_child = pose_3d[:, idx, :]
        rest_parent = rest_pose_3d[:, parent, :]
        # import pdb; pdb.set_trace()
        rest_child = rest_pose_3d[:, idx, :]
        axis_ang = ik_hand_pose(p_parent, p_child, rest_parent, rest_child)
        axis_ang_list[:, i, :] = axis_ang
    for i, idx in enumerate(exclude_children):
        # print("i = ", i, "idx = ", idx)
        parent = JOINT_PARENT_ID_DICT[idx]
        if parent == -1:
            continue
        # print(pose_3d.shape, rest_pose_3d.shape)
        p_parent = pose_3d[:, parent, :]
        p_child = pose_3d[:, idx, :]
        rest_parent = rest_pose_3d[:, parent, :]
        rest_child = rest_pose_3d[:, idx, :]
        axis_ang = ik_hand_pose(p_parent, p_child, rest_parent, rest_child)
        exclude_axis_ang_list[:, i, :] = axis_ang
    # 从全局变换计算局部local变换
    axis_ang_local_list = torch.zeros_like(axis_ang_list)
    for i in range(len(used_children)):
        child_idx = used_children[i]
        parent = JOINT_PARENT_ID_DICT[child_idx]
        if parent == -1:
            continue
        if parent in exclude_children:
            parent_i = exclude_children.index(parent)
            parent_axis_ang = exclude_axis_ang_list[:, parent_i, :]
            child_axis_ang = axis_ang_list[:, i, :]
        elif parent in used_children:
            parent_i = used_children.index(parent)
            parent_axis_ang = axis_ang_list[:, parent_i, :]
            child_axis_ang = axis_ang_list[:, i, :]
            
        # 转为四元数和旋转矩阵
        parent_quat = batch_aa2quat(parent_axis_ang)
        child_quat = batch_aa2quat(child_axis_ang)
        parent_rot = quat2mat(parent_quat)
        child_rot = quat2mat(child_quat)
        
        # 计算局部旋转: local = parent^T * child
        local_rot = torch.matmul(parent_rot.transpose(1, 2), child_rot)
        
        # 转回轴角: 旋转矩阵 -> 四元数 -> 轴角
        from pytorch3d.transforms import matrix_to_axis_angle
        local_axis_ang = matrix_to_axis_angle(local_rot)
        # print("local_axis_ang.shape = ", local_axis_ang.shape)
        axis_ang_local_list[:, i, :] = local_axis_ang

    # pose_3d = pose_3d.reshape(pose_3d.shape[0], -1)
    # import pdb; pdb.set_trace()
    length = torch.norm(axis_ang_local_list, dim=-1, keepdim=True)
    length[length < 1e-8] = 1.0
    direction = axis_ang_local_list / length
    # print("axis_ang_local_list: ", axis_ang_local_list)
    return axis_ang_local_list, direction, length

def ik_hand_pose(p_parent, p_child, rest_parent, rest_child):
    # 输入都是 B x 3
    v_pose = p_child - p_parent  # B x 3
    v_rest = rest_child - rest_parent  # B x 3
    v_pose = v_pose / (torch.norm(v_pose, dim=-1, keepdim=True) + 1e-8)
    v_rest = v_rest / (torch.norm(v_rest, dim=-1, keepdim=True) + 1e-8)

    cos_theta = torch.sum(v_pose * v_rest, dim=-1, keepdim=True)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    axis = torch.cross(v_rest, v_pose, dim=-1)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    axis = torch.nan_to_num(axis, nan=0.0, posinf=0.0, neginf=0.0)
    axis_ang = axis * theta
    axis_ang = torch.nan_to_num(axis_ang, nan=0.0, posinf=0.0, neginf=0.0)
    return axis_ang



def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
        rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results

def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res

def th_scalemat_scale(th_scale_bone):
    batch_size = th_scale_bone.shape[0]
    th_scale_bone_mat = torch.eye(4).repeat([batch_size, th_scale_bone.shape[1], 1, 1])
    th_scale_bone_mat = th_scale_bone_mat.type_as(th_scale_bone).to(th_scale_bone.device)
    if len(th_scale_bone.shape) == 3:
        for s in range(th_scale_bone.shape[1]):
            th_scale_bone_mat[:, s, 0, 0] = th_scale_bone[:, s, 0]
            th_scale_bone_mat[:, s, 1, 1] = th_scale_bone[:, s, 1]
            th_scale_bone_mat[:, s, 2, 2] = th_scale_bone[:, s, 2]
    else:
        for s in range(th_scale_bone.shape[1]):
            th_scale_bone_mat[:, s, 0, 0] = th_scale_bone[:, s]
            th_scale_bone_mat[:, s, 1, 1] = th_scale_bone[:, s]
            th_scale_bone_mat[:, s, 2, 2] = th_scale_bone[:, s] 
    return th_scale_bone_mat


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res



def vertices2landmarks(
    vertices,
    faces,
    lmk_faces_idx,
    lmk_bary_coords
):
    ''' 
        Calculates landmarks by barycentric interpolation
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
        
        Modified from https://github.com/vchoutas/smplx
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    # lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        # batch_size, -1, 3)
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        1, -1, 3)
    lmk_faces = lmk_faces.repeat([batch_size,1,1])

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.reshape(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,lf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks



def save_textured_nimble(fname, skin_v, tex_img):
    # 保持你的目录结构不变
    import cv2
    textured_pkl = "assets/NIMBLE_TEX_FUV.pkl"
    from pathlib import Path
    fname = Path(fname)

    # 你的结构: skin/skin/xxx.obj, skin/texture/xxx_diffuse.png, skin/mtl/xxx.mtl
    skin_dir = fname.parent
    tex_dir = skin_dir.parent / "texture"
    normal_dir = skin_dir.parent / "normal"
    spec_dir = skin_dir.parent / "spec"
    mtl_dir = skin_dir.parent / "mtl"

    skin_dir.mkdir(exist_ok=True)
    tex_dir.mkdir(exist_ok=True)
    normal_dir.mkdir(exist_ok=True)
    spec_dir.mkdir(exist_ok=True)
    mtl_dir.mkdir(exist_ok=True)

    obj_name_skin = fname
    mtl_name = mtl_dir / (fname.stem + ".mtl")
    tex_name_diffuse = tex_dir / (fname.stem + "_diffuse.png")
    tex_img = np.uint8(tex_img * 255)
    cv2.imwrite(str(tex_name_diffuse), tex_img[:, :, :3])
    cv2.imwrite(str(normal_dir / (fname.stem + "_normal.png")), tex_img[:, :, 3:6])
    cv2.imwrite(str(spec_dir / (fname.stem + "_spec.png")), tex_img[:, :, 6:])
    
    # mtl: 纹理路径写成 ../../texture/xxx_diffuse.png
    mtl_str = "newmtl material_0\nKa 0.200000 0.200000 0.200000\nKd 0.800000 0.800000 0.800000\nKs 1.000000 1.000000 1.000000\nTr 1.000000\nillum 2\nNs 0.000000\nmap_Kd "
    tex_rel_path = os.path.relpath(tex_name_diffuse, start=skin_dir)
    mtl_str = mtl_str + tex_rel_path.replace("\\", "/")
    with open(mtl_name, "w") as f:
        f.writelines(mtl_str)
    print("save to", mtl_name)

    # obj
    f_uv = np.load(textured_pkl, allow_pickle=True)
    with open(obj_name_skin, "w") as f:
        # 写入纹理
        f.writelines("mtllib ../mtl/{}.mtl\n".format(fname.stem))
        # 写入材质
        f.writelines("usemtl material_0\n")

        for v in skin_v:
            f.writelines("v {:.5f} {:.5f} {:.5f}\n".format(v[0], v[1], v[2]))
        f.writelines(f_uv)
    print("save to", fname)




def smooth_mesh(mesh_p3d):
    mesh_p3d_smooth = pytorch3d.ops.mesh_filtering.taubin_smoothing(mesh_p3d, num_iter=3)
    target_mv = mesh_p3d_smooth.verts_padded()
    nan_mv = torch.isnan(target_mv)
    target_mv[nan_mv] = mesh_p3d.verts_padded()[nan_mv]  
    mesh_p3d_smooth_fixnan = Meshes(target_mv, mesh_p3d.faces_padded())
    return mesh_p3d_smooth_fixnan

def axisang_to_local_rot(axis_ang_list, used_children, exclude_children):
    # axis_ang_list: [B, N, 3]
    quats = batch_aa2quat(axis_ang_list.view(-1, 3))  # [B*N, 4]
    rot_mats = quat2mat(quats).view(axis_ang_list.shape[0], axis_ang_list.shape[1], 3, 3)  # [B, N, 3, 3]
    local_rot_mats = torch.zeros_like(rot_mats)
    for i, idx in enumerate(used_children):
        parent = JOINT_PARENT_ID_DICT[idx]
        if parent == -1 or parent in exclude_children:
            local_rot_mats[:, i] = rot_mats[:, i]
        else:
            parent_i = used_children.index(parent)
            parent_rot = rot_mats[:, parent_i]
            child_rot = rot_mats[:, i]
            local_rot_mats[:, i] = torch.matmul(parent_rot.transpose(1, 2), child_rot)
    return local_rot_mats
