import xml.etree.ElementTree as ET
import numpy as np
import json

def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix.
    Args:
        quat (list): [w, x, y, z]
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # 确保四元数的长度为1
    quat = np.array(quat)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    return R

def parse_camera_pose_and_intrinsics(xml_file, image_width=54, image_height=42):
    """
    Parse camera poses and intrinsics from MuJoCo XML file.
    Args:
        xml_file (str): Path to the MuJoCo XML file.
        image_width (int): Image width.
        image_height (int): Image height.
    Returns:
        dict: Dictionary containing camera poses and intrinsics {camera_name: {R, T, K, R_world_to_hand, T_world_to_hand}}
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    camera_data = {}
    old_cam_data = {}
    hand_data = {}

    # 获取手部的位姿
    hand_body = root.find(".//body[@name='hand']")
    if hand_body is not None:
        hand_pos = np.array([float(x) for x in hand_body.get("pos").split()])
        hand_quat = [float(x) for x in hand_body.get("quat").split()]
        R_world_to_hand = quaternion_to_rotation_matrix(hand_quat)
        T_world_to_hand = np.array([float(x) for x in hand_body.get("pos").split()])
        R_hand_to_world = R_world_to_hand.T  # 转置得到手部到世界的旋转矩阵
        T_hand_to_world = -np.dot(R_hand_to_world, T_world_to_hand)
        hand_data = {
            "R_world_to_hand": R_world_to_hand.tolist(),
            "T_world_to_hand": T_world_to_hand.tolist(),
            "R_hand_to_world": R_hand_to_world.tolist(),
            "T_hand_to_world": T_hand_to_world.tolist()
        }
        old_cam_data['hand'] = hand_data
    else:
        raise ValueError("Hand body not found in XML file.")

    # Iterate through all camera elements
    for camera in root.findall(".//camera"):
        name = camera.get("name")
        pos = camera.get("pos")
        quat = camera.get("quat")
        fovy = camera.get("fovy")

        if pos and quat and fovy:
            # Parse position (T)
            T = np.array([float(x) for x in pos.split()])

            # Parse quaternion and convert to rotation matrix (R)
            quat = [float(x) for x in quat.split()]
            R = quaternion_to_rotation_matrix(quat)
            R = np.dot(R, np.diag([1, 1, -1]))  # 调整坐标系

            # Compute world-to-camera transformation
            R_world_to_camera = R.T  # Transpose of R
            T_world_to_camera = -np.dot(R_world_to_camera, T)  # Negative dot product

            # Parse field of view (fovy) and calculate intrinsic matrix (K)
            fovy = float(fovy)
            f_y = image_height / (2 * np.tan(np.radians(fovy) / 2))
            f_x = f_y * (image_width / image_height)
            c_x = image_width / 2
            c_y = image_height / 2
            K = np.array([
                [f_x, 0, c_x],
                [0, f_y, c_y],
                [0, 0, 1]
            ])

            # Store the data
            camera_data[name] = {
                "R": R_world_to_camera.tolist(),
                "T": T_world_to_camera.tolist(),
                "K": K.tolist(),
            }
            old_cam_data[name] = {
                "R": R.tolist(),
                "T": T.tolist(),
                "K": K.tolist()
            }

    return camera_data, old_cam_data

def save_camera_data_to_json(camera_data, output_file):
    """
    Save camera data to JSON file.
    Args:
        camera_data (dict): Camera data {camera_name: {R, T, K}}
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(camera_data, f, indent=4)
    print(f"Camera data saved to {output_file}")

# 示例使用
if __name__ == "__main__":
    xml_file = "/home/zhangrong/worker_lingyu/zhangrong/HandPose/NIMBLE_model/models_mano.xml"
    output_json = "camera_poses_world_to_camera.json"

    camera_data, old_cam_data = parse_camera_pose_and_intrinsics(xml_file)
    save_camera_data_to_json(camera_data, output_json)
    save_camera_data_to_json(old_cam_data, "camera_poses_old.json")