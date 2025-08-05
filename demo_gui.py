import sys
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGridLayout
from PyQt5.QtCore import Qt
import open3d as o3d
from NIMBLELayer import NIMBLELayer
import os
from utils import *

from PyQt5.QtCore import QTimer

class PoseControlWidget(QWidget):
    def __init__(self, nlayer, shape_param, device):
        super().__init__()
        self.nlayer = nlayer
        self.shape_param = shape_param
        self.device = device
        self.num_joints = 20
        self.direction = np.zeros((self.num_joints, 3))  # 方向向量
        self.length = np.ones(self.num_joints)            # 模长
        self.current_joint = 0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hand Pose Control GUI")
        layout = QVBoxLayout()

        # --- 新增：嵌入Open3D Widget（如果可用） ---
        try:
            import open3d.visualization.gui as o3d_gui
            self.o3d_widget = o3d_gui.O3DWidget()
            layout.addWidget(self.o3d_widget)
            self.use_o3d_widget = True
        except Exception:
            self.o3d_widget = None
            self.use_o3d_widget = False
        # --- end ---

        self.grid = QGridLayout()
        self.dir_edits = []
        self.len_edits = []
        for i in range(self.num_joints):
            self.grid.addWidget(QLabel(f"Joint {i+1} Dir(x,y,z):"), i, 0)
            dir_edit = QLineEdit("0,0,0")
            self.grid.addWidget(dir_edit, i, 1)
            self.dir_edits.append(dir_edit)
            self.grid.addWidget(QLabel("Len:"), i, 2)
            len_edit = QLineEdit("1")
            self.grid.addWidget(len_edit, i, 3)
            self.len_edits.append(len_edit)
        layout.addLayout(self.grid)

        self.info_label = QLabel("Use ↑↓←→ to change direction, +/- to change length, Enter to confirm, Esc to quit. PageUp/PageDown切换关节")
        layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        self.next_btn = QPushButton("Show 3D")
        self.next_btn.clicked.connect(self.show_3d)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        # 键盘控制当前关节的方向和长度
        if event.key() == Qt.Key_Up:
            self.direction[self.current_joint][1] += 0.1
        elif event.key() == Qt.Key_Down:
            self.direction[self.current_joint][1] -= 0.1
        elif event.key() == Qt.Key_Left:
            self.direction[self.current_joint][0] -= 0.1
        elif event.key() == Qt.Key_Right:
            self.direction[self.current_joint][0] += 0.1
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.length[self.current_joint] += 0.1
        elif event.key() == Qt.Key_Minus:
            self.length[self.current_joint] = max(0.1, self.length[self.current_joint] - 0.1)
        elif event.key() == Qt.Key_PageUp:
            self.current_joint = (self.current_joint + 1) % self.num_joints
        elif event.key() == Qt.Key_PageDown:
            self.current_joint = (self.current_joint - 1) % self.num_joints
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.show_3d()
        elif event.key() == Qt.Key_Escape:
            self.close()
        self.update_edits()

    def update_edits(self):
        for i in range(self.num_joints):
            self.dir_edits[i].setText(",".join([f"{v:.2f}" for v in self.direction[i]]))
            self.len_edits[i].setText(f"{self.length[i]:.2f}")

    def get_pose_param(self):
        pose_param = np.zeros((self.num_joints, 3))
        for i in range(self.num_joints):
            # 方向归一化后乘以长度
            dir_vec = self.direction[i]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                pose_param[i] = dir_vec / norm * self.length[i]
            else:
                pose_param[i] = np.zeros(3)
        return torch.tensor(pose_param, dtype=torch.float32, device=self.device).view(1, -1)

# ...existing code...

    def show_3d(self):
        for i in range(self.num_joints):
            dir_str = self.dir_edits[i].text()
            dir_vals = [float(x) for x in dir_str.split(",")]
            self.direction[i] = dir_vals
            self.length[i] = float(self.len_edits[i].text())
        pose_param = self.get_pose_param()
        joints = self.nlayer.forward_joints(pose_param, self.shape_param, None, None)[0].detach().cpu().numpy()
        lines = []
        for joint_id, parent_id in JOINT_PARENT_ID_DICT.items():
            if parent_id != -1:
                lines.append([joint_id, parent_id])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(joints)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(lines), 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(joints)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (joints.shape[0], 1)))
        # --- 新增：嵌入Open3D Widget或弹窗 ---
        if hasattr(self, 'use_o3d_widget') and self.use_o3d_widget and self.o3d_widget is not None:
            self.o3d_widget.scene.clear_geometry()
            self.o3d_widget.scene.add_geometry("skeleton", line_set, o3d.visualization.rendering.MaterialRecord())
            self.o3d_widget.scene.add_geometry("joints", pcd, o3d.visualization.rendering.MaterialRecord())
        else:
            # 用QTimer让draw_geometries在事件循环后执行，避免卡死
            def show_popup():
                o3d.visualization.draw_geometries([line_set, pcd])
            QTimer.singleShot(10, show_popup)
        # --- end ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化NIMBLELayer和参数
    pm_dict_name = r"assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"assets/NIMBLE_TEX_DICT.pkl"
    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)
    else:
        pm_dict = None
    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)
    else:
        tex_dict = None
    if os.path.exists(r"assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    else:
        nimble_mano_vreg = None

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=False, pose_ncomp=60, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    shape_param = torch.zeros(1, 20, device=device)
    app = QApplication(sys.argv)
    w = PoseControlWidget(nlayer, shape_param, device)
    w.show()
    sys.exit(app.exec_())