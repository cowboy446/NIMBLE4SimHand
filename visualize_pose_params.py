import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes

def visualize_pose_basis_effect(nlayer, output_folder='pose_basis_visualization'):
    """
    Visualizes the effect of each principal component in the pose basis.
    Creates visualizations showing how each component affects joint rotation.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the pose basis and related parameters
    pose_basis = nlayer.pose_basis.detach().cpu().numpy()  # Shape: [pose_dims, pose_ncomp]
    pose_mean = nlayer.pose_mean.detach().cpu().numpy()    # Shape: [pose_dims]
    
    print(f"Pose basis shape: {pose_basis.shape}")
    print(f"Pose mean shape: {pose_mean.shape}")
    
    # Number of joints (each joint has 3 rotation parameters)
    joint_dims = pose_basis.shape[0] // 3
    pose_ncomp = pose_basis.shape[1]
    
    # Create a visualization of each component's effect on joint rotations
    for comp_idx in range(min(30, pose_ncomp)):
        component = pose_basis[:, comp_idx]
        
        # Calculate the effect on each joint (Euclidean norm of 3D rotation params)
        joint_effects = np.zeros(joint_dims)
        for j in range(joint_dims):
            joint_effects[j] = np.linalg.norm(component[j*3:(j+1)*3])
        
        # Plot the joint effects
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(joint_dims), joint_effects)
        
        # Add labels
        plt.xlabel('Joint Index')
        plt.ylabel('Effect Magnitude')
        plt.title(f'Component {comp_idx} Effect on Joint Rotations')
        plt.xticks(range(joint_dims))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add joint names if available
        if hasattr(nlayer, 'joint_names'):
            plt.xticks(range(joint_dims), nlayer.joint_names, rotation=45)
        
        # Highlight the top affected joints
        top_joints = np.argsort(-joint_effects)[:3]
        for j in top_joints:
            bars[j].set_color('red')
            plt.text(j, joint_effects[j] + 0.01, f'Joint {j}', ha='center')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'component_{comp_idx}_joint_effects.png'))
        plt.close()
        
        print(f"Generated visualization for component {comp_idx}")

def generate_pose_variation_grid(nlayer, comp_idx=0, values=[-3.0, -1.5, 0, 1.5, 3.0], output_folder='pose_variations'):
    """
    Generates a grid of hand poses by varying a specific pose component.
    
    Args:
        nlayer: NIMBLE layer model
        comp_idx: Index of the PCA component to vary
        values: List of values to use for the component
        output_folder: Output folder for the generated meshes
    """
    device = nlayer.device
    os.makedirs(output_folder, exist_ok=True)
    
    # Use average hand shape and texture
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    for value in values:
        # Create pose parameters with only one component active
        pose_param = torch.zeros(1, 30, device=device)
        pose_param[0, comp_idx] = value
        
        # Generate hand mesh
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(
            pose_param, shape_param, tex_param, handle_collision=True
        )
        
        # Create and smooth mesh
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        # Save as OBJ
        output_file = os.path.join(output_folder, f"component_{comp_idx}_value_{value:.2f}.obj")
        pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
        
        print(f"Generated mesh for component {comp_idx}, value {value}")

def visualize_pose_correlation_matrix(nlayer, output_folder='pose_analysis'):
    """
    Generates a correlation matrix between the pose basis components.
    This helps to understand relationships between different PCA components.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the pose basis
    pose_basis = nlayer.pose_basis.detach().cpu().numpy()  # Shape: [pose_dims, pose_ncomp]
    pose_ncomp = pose_basis.shape[1]
    
    # Compute correlation between components
    corr_matrix = np.zeros((pose_ncomp, pose_ncomp))
    for i in range(pose_ncomp):
        for j in range(pose_ncomp):
            # Compute correlation as cosine similarity
            corr = np.dot(pose_basis[:, i], pose_basis[:, j]) / (
                np.linalg.norm(pose_basis[:, i]) * np.linalg.norm(pose_basis[:, j]))
            corr_matrix[i, j] = corr
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Correlation Between Pose Basis Components')
    plt.xlabel('Component Index')
    plt.ylabel('Component Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pose_component_correlation.png'))
    plt.close()
    
    print(f"Generated pose component correlation matrix")

def generate_finger_specific_poses(nlayer, output_folder='finger_poses'):
    """
    Attempts to generate poses where each finger is individually activated.
    This helps to identify which components control which fingers.
    """
    device = nlayer.device
    os.makedirs(output_folder, exist_ok=True)
    
    # Use average hand shape and texture
    shape_param = torch.zeros(1, 20, device=device)
    tex_param = torch.zeros(1, 10, device=device)
    
    # Dictionary to store component indices for each finger
    finger_components = {
        'thumb': [0, 1, 2, 3],       # Components likely affecting the thumb
        'index': [4, 5, 6, 7],       # Components likely affecting the index finger
        'middle': [8, 9, 10, 11],    # Components likely affecting the middle finger
        'ring': [12, 13, 14, 15],    # Components likely affecting the ring finger
        'pinky': [16, 17, 18, 19]    # Components likely affecting the pinky finger
    }
    
    # Try different combinations for each finger
    for finger, components in finger_components.items():
        for i, comp_idx in enumerate(components):
            # Test each component separately
            pose_param = torch.zeros(1, 30, device=device)
            pose_param[0, comp_idx] = 2.0  # Strong activation
            
            # Generate hand mesh
            skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(
                pose_param, shape_param, tex_param, handle_collision=True
            )
            
            # Create and smooth mesh
            skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
            skin_p3dmesh = smooth_mesh(skin_p3dmesh)
            
            # Save as OBJ
            output_file = os.path.join(output_folder, f"{finger}_comp_{comp_idx}.obj")
            pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
            
            print(f"Generated potential {finger} pose using component {comp_idx}")
        
        # Also try combining components for more realistic finger poses
        pose_param = torch.zeros(1, 30, device=device)
        for comp_idx in components:
            pose_param[0, comp_idx] = 1.0  # Combined activation
            
        # Generate hand mesh
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(
            pose_param, shape_param, tex_param, handle_collision=True
        )
        
        # Create and smooth mesh
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(1, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        # Save as OBJ
        output_file = os.path.join(output_folder, f"{finger}_combined.obj")
        pytorch3d.io.IO().save_mesh(skin_p3dmesh[0], output_file)
        
        print(f"Generated combined {finger} pose")

if __name__ == "__main__":
    # Set up the NIMBLE model
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
    
    print("1. Visualizing pose basis effects...")
    visualize_pose_basis_effect(nlayer)
    
    print("\n2. Generating variations for top components...")
    # Generate variations for the top 5 components
    for i in range(5):
        generate_pose_variation_grid(nlayer, comp_idx=i)
    
    print("\n3. Generating correlation matrix...")
    visualize_pose_correlation_matrix(nlayer)
    
    print("\n4. Attempting to isolate finger-specific poses...")
    generate_finger_specific_poses(nlayer)
    
    print("\nAll visualizations and analyses complete!")
