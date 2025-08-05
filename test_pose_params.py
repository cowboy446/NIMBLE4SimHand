import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes

def test_individual_pose_parameters():
    """
    Test each pose parameter individually to see its effect on the hand mesh.
    This helps understand what each of the 30 parameters controls.
    """
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
    
    # Create output folder
    output_folder = "individual_params"
    os.makedirs(output_folder, exist_ok=True)
    
    # Use average hand shape and texture
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    # Test each parameter with different values
    test_values = [-2.0, -1.0, 1.0, 2.0]
    
    # First generate the neutral pose as reference
    pose_param = torch.zeros(1, 30, device=device)
    skin_v, _, _, _, _ = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
    skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
    skin_p3dmesh = smooth_mesh(skin_p3dmesh)
    pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], os.path.join(output_folder, "neutral_pose.obj"))
    print("Generated neutral pose reference")
    
    # Test each parameter individually
    for i in range(30):
        for value in test_values:
            # Create pose parameters with only one component active
            pose_param = torch.zeros(1, 30, device=device)
            pose_param[0, i] = value
            
            # Generate hand mesh
            skin_v, _, _, _, _ = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
            
            # Create and smooth mesh
            skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
            skin_p3dmesh = smooth_mesh(skin_p3dmesh)
            
            # Save as OBJ
            output_file = os.path.join(output_folder, f"param_{i}_value_{value}.obj")
            pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
            
            print(f"Generated mesh for parameter {i}, value {value}")
            
        # Clean up GPU memory
        torch.cuda.empty_cache()

def generate_number_gestures():
    """
    Generate hand gestures for numbers 1-10 based on our understanding
    of the pose parameters from the individual parameter tests.
    """
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
    
    # Create output folder
    output_folder = "number_gestures_refined"
    os.makedirs(output_folder, exist_ok=True)
    
    # Use average hand shape and texture
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    # Define hand gesture poses for numbers 1-10
    # These values should be refined after testing individual parameters
    number_gestures = {
        1: {  # Index finger extended
            0: 2.0,   # Primary index finger component
            5: -1.5,  # Curl other fingers
            10: -1.5, # Curl other fingers
            15: -1.5  # Curl other fingers
        },
        2: {  # Index and middle fingers extended
            0: 2.0,   # Index finger component
            5: 2.0,   # Middle finger component
            10: -1.5, # Curl other fingers
            15: -1.5  # Curl other fingers
        },
        3: {  # Index, middle, and ring fingers extended
            0: 2.0,   # Index finger component
            5: 2.0,   # Middle finger component
            10: 2.0,  # Ring finger component
            15: -1.5  # Curl pinky
        },
        4: {  # Index, middle, ring, and pinky extended
            0: 2.0,   # Index finger component
            5: 2.0,   # Middle finger component
            10: 2.0,  # Ring finger component
            15: 2.0   # Pinky component
        },
        5: {  # All fingers extended
            0: 2.0,   # Index finger component
            2: 2.0,   # Thumb component
            5: 2.0,   # Middle finger component
            10: 2.0,  # Ring finger component
            15: 2.0   # Pinky component
        },
        # Additional gestures 6-10 will be defined based on testing results
    }
    
    # Generate each number gesture
    for number, params in number_gestures.items():
        # Create pose parameter tensor
        pose_param = torch.zeros(1, 30, device=device)
        
        # Set the parameters for this gesture
        for idx, value in params.items():
            pose_param[0, idx] = value
        
        # Generate hand mesh
        skin_v, _, _, _, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
        
        # Create and smooth mesh
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        # Save the mesh
        output_file = os.path.join(output_folder, f"number_{number}.obj")
        pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
        
        # Also save parameters for reference
        param_file = os.path.join(output_folder, f"number_{number}_params.txt")
        with open(param_file, 'w') as f:
            f.write(f"Pose parameters for number {number}:\n")
            for i in range(30):
                f.write(f"Param {i}: {pose_param[0, i].item():.4f}\n")
        
        print(f"Generated hand gesture for number {number}")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Testing individual pose parameters...")
    test_individual_pose_parameters()
    
    print("\nGenerating number gestures based on parameter understanding...")
    generate_number_gestures()
