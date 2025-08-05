import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes

def print_pose_basis_info(nlayer):
    """Print information about the pose basis to understand the PCA components"""
    pose_basis = nlayer.pose_basis.detach().cpu().numpy()
    pose_mean = nlayer.pose_mean.detach().cpu().numpy()
    pose_pm_std = nlayer.pose_pm_std.detach().cpu().numpy()
    pose_pm_mean = nlayer.pose_pm_mean.detach().cpu().numpy()
    
    print(f"Pose basis shape: {pose_basis.shape}")
    print(f"Pose mean shape: {pose_mean.shape}")
    print(f"Pose PM std shape: {pose_pm_std.shape}")
    print(f"Pose PM mean shape: {pose_pm_mean.shape}")
    
    # Analyze the principal components
    component_norms = np.linalg.norm(pose_basis, axis=0)
    print("\nTop 10 principal components by norm:")
    top_indices = np.argsort(-component_norms)[:10]
    for i, idx in enumerate(top_indices):
        print(f"Component {idx}: Norm = {component_norms[idx]:.4f}")
    
    # See which joints are most affected by each component
    full_pose_dims = pose_basis.shape[0]
    joint_dims = full_pose_dims // 3  # Each joint has 3 rotation parameters
    
    print("\nMost affected joints per component:")
    for comp_idx in range(min(10, pose_basis.shape[1])):
        component = pose_basis[:, comp_idx]
        # Reshape to get per-joint effect
        joint_effect = np.zeros(joint_dims)
        for j in range(joint_dims):
            joint_effect[j] = np.linalg.norm(component[j*3:(j+1)*3])
        
        # Get top affected joints
        top_joints = np.argsort(-joint_effect)[:3]
        print(f"Component {comp_idx}: Top joints = {top_joints}, Effects = {joint_effect[top_joints]}")

def generate_hand_gestures_for_numbers():
    """Generate hand gestures for numbers 1-10 by manipulating pose parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model data
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
        nimble_mano_vreg = None

    # Create NIMBLE layer
    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, 
                         pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    
    # Print information about the pose basis
    print_pose_basis_info(nlayer)
    
    # Create output folder
    output_folder = "number_gestures"
    os.makedirs(output_folder, exist_ok=True)
    
    # Use average hand shape and texture
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    # Define specific pose parameters for each number gesture
    # These values should be adjusted based on experimentation
    # The following is just a starting point based on understanding of hand anatomy
    number_poses = {
        # Number 1: Index finger extended, others curled
        1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, -1.5, 0.0, -1.5, 0.0,
                     -1.5, 0.0, -1.5, 0.0, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 2: Index and middle fingers extended
        2: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, -1.5, 1.5, -1.5, 0.0,
                     -1.5, 0.0, -1.5, 0.0, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 3: Index, middle, and ring fingers extended
        3: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, -1.5, 1.5, -1.5, 1.5,
                     -1.5, 0.0, -1.5, 0.0, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 4: All fingers except thumb extended
        4: np.array([0.0, 0.0, 0.0, 0.0, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5,
                     -1.5, 1.5, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 5: All fingers extended
        5: np.array([0.0, 0.0, 0.0, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5,
                     -1.5, 1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 6: Thumb and pinky curled, rest extended (Western style)
        6: np.array([0.0, 0.0, 0.0, -1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5,
                     -1.5, -1.5, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 7: Thumb, index and middle extended (Western style)
        7: np.array([0.0, 0.0, 0.0, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5, -1.5,
                     -1.5, -1.5, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 8: Thumb, index, middle extended, slight separation (Western style)
        8: np.array([0.0, 0.0, 0.0, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5, -1.5,
                     -1.5, -1.5, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.5,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 9: All fingers curled except index (Western style)
        9: np.array([0.0, 0.0, 0.0, -1.5, -1.5, 1.5, -1.5, -1.5, -1.5, -1.5,
                     -1.5, -1.5, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Number 10: Number 1 + Number 0 (fist with extended thumb)
        10: np.array([0.0, 0.0, 0.0, 1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5,
                      -1.5, -1.5, -1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    
    # Generate and save each number gesture
    for number, pose_values in number_poses.items():
        # Create pose parameter tensor
        pose_param = torch.tensor(pose_values, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Generate hand mesh
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(
            pose_param, shape_param, tex_param, handle_collision=True
        )
        
        # Create and smooth mesh
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        # Save the mesh as OBJ
        output_file = os.path.join(output_folder, f"number_{number}.obj")
        pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
        
        # Also save with texture
        tex_img_np = tex_img.detach().cpu().numpy()
        skin_v_np = skin_p3dmesh.verts_padded().detach().cpu().numpy()
        save_textured_nimble(
            os.path.join(output_folder, f"number_{number}_textured.obj"), 
            skin_v_np[0], 
            tex_img_np[0]
        )
        
        print(f"Generated and saved gesture for number {number}")
    
    print("All number gestures generated successfully!")

# Additional function to refine the gestures after initial experimentation
def interactive_pose_adjustment():
    """Interactive tool to adjust pose parameters and see the results"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model data
    pm_dict_name = r"assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"assets/NIMBLE_TEX_DICT.pkl"
    nimble_mano_vreg = None
    
    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

    if os.path.exists(r"assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    
    # Create NIMBLE layer
    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, 
                         pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    
    # Create output folder
    output_folder = "pose_adjustments"
    os.makedirs(output_folder, exist_ok=True)
    
    # Use average hand shape and texture
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    # Starting pose - all zeros (neutral pose)
    pose_values = np.zeros(30)
    iteration = 0
    
    while True:
        # Show current pose parameters
        print(f"\nCurrent pose parameters (iteration {iteration}):")
        for i in range(0, 30, 5):
            end = min(i+5, 30)
            params_str = " ".join([f"{pose_values[j]:.2f}" for j in range(i, end)])
            print(f"Params {i}-{end-1}: {params_str}")
        
        # Generate mesh with current parameters
        pose_param = torch.tensor(pose_values, dtype=torch.float32, device=device).unsqueeze(0)
        skin_v, _, _, _, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
        
        # Create and smooth mesh
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        # Save current iteration
        output_file = os.path.join(output_folder, f"iteration_{iteration}.obj")
        pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
        
        print(f"\nSaved mesh at: {output_file}")
        
        # Get user input for adjustment
        print("\nOptions:")
        print("1. Adjust parameter value")
        print("2. Save current pose with a name")
        print("3. Load a predefined number pose")
        print("4. Quit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == "1":
            param_idx = int(input("Enter parameter index (0-29): "))
            param_val = float(input("Enter new value: "))
            pose_values[param_idx] = param_val
        
        elif choice == "2":
            pose_name = input("Enter name for this pose: ")
            output_file = os.path.join(output_folder, f"{pose_name}.obj")
            pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
            
            # Also save the parameter values for future reference
            param_file = os.path.join(output_folder, f"{pose_name}_params.txt")
            np.savetxt(param_file, pose_values)
            
            print(f"Saved pose as {output_file} with parameters in {param_file}")
        
        elif choice == "3":
            number = int(input("Enter number (1-10): "))
            # This assumes generate_hand_gestures_for_numbers has been run previously
            param_file = os.path.join("number_gestures", f"number_{number}_params.txt")
            
            if os.path.exists(param_file):
                pose_values = np.loadtxt(param_file)
                print(f"Loaded parameters for number {number}")
            else:
                print(f"No saved parameters found for number {number}. Run generate_hand_gestures_for_numbers first.")
        
        elif choice == "4":
            print("Exiting interactive mode.")
            break
        
        else:
            print("Invalid choice, please try again.")
        
        iteration += 1

if __name__ == "__main__":
    print("Generating hand gestures for numbers 1-10...")
    generate_hand_gestures_for_numbers()
    
    # Uncomment to use the interactive adjustment tool after initial generation
    # print("\nStarting interactive pose adjustment tool...")
    # interactive_pose_adjustment()
