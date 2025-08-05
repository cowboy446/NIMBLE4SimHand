import os
import torch
import numpy as np
from NIMBLELayer import NIMBLELayer

from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes

class HandPoseGenerator:
    """A class to analyze and generate hand poses using the NIMBLE model"""
    
    def __init__(self, device=None):
        """Initialize the NIMBLE model and related resources"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model data
        pm_dict_name = r"assets/NIMBLE_DICT_9137.pkl"
        tex_dict_name = r"assets/NIMBLE_TEX_DICT.pkl"
        vreg_path = r"assets/NIMBLE_MANO_VREG.pkl"

        if os.path.exists(pm_dict_name):
            pm_dict = np.load(pm_dict_name, allow_pickle=True)
            self.pm_dict = batch_to_tensor_device(pm_dict, self.device)
        else:
            raise FileNotFoundError(f"Could not find {pm_dict_name}")

        if os.path.exists(tex_dict_name):
            tex_dict = np.load(tex_dict_name, allow_pickle=True)
            self.tex_dict = batch_to_tensor_device(tex_dict, self.device)
        else:
            raise FileNotFoundError(f"Could not find {tex_dict_name}")

        if os.path.exists(vreg_path):
            nimble_mano_vreg = np.load(vreg_path, allow_pickle=True)
            self.nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, self.device)
        else:
            self.nimble_mano_vreg = None
            print("Warning: NIMBLE_MANO_VREG.pkl not found. Some functionality may be limited.")

        # Create NIMBLE layer
        self.nlayer = NIMBLELayer(
            self.pm_dict, self.tex_dict, self.device, 
            use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, 
            nimble_mano_vreg=self.nimble_mano_vreg
        )
        
        # Default parameters
        self.default_shape = torch.zeros(1, 20, device=self.device)
        self.default_texture = torch.zeros(1, 10, device=self.device)
        
        # Store information about the pose basis
        self._analyze_pose_basis()
    
    def _analyze_pose_basis(self):
        """Analyze the pose basis to understand component effects"""
        pose_basis = self.nlayer.pose_basis.detach().cpu().numpy()
        self.pose_dims, self.pose_ncomp = pose_basis.shape
        self.joint_count = self.pose_dims // 3
        
        # Calculate effect of each component on each joint
        self.joint_effects = np.zeros((self.pose_ncomp, self.joint_count))
        for comp_idx in range(self.pose_ncomp):
            for joint_idx in range(self.joint_count):
                # Calculate the norm of the effect on this joint's rotation
                start_idx = joint_idx * 3
                end_idx = start_idx + 3
                self.joint_effects[comp_idx, joint_idx] = np.linalg.norm(
                    pose_basis[start_idx:end_idx, comp_idx]
                )
        
        # For each component, find the top affected joints
        self.top_joints_per_component = {}
        for comp_idx in range(self.pose_ncomp):
            # Get the indices of the top 3 affected joints
            top_joints = np.argsort(-self.joint_effects[comp_idx])[:3]
            self.top_joints_per_component[comp_idx] = {
                'joints': top_joints.tolist(),
                'effects': self.joint_effects[comp_idx, top_joints].tolist()
            }
        
        print("Pose basis analysis complete. See joint effects with get_component_info()")
    
    def get_component_info(self, component_idx=None):
        """Get information about the effect of pose components on joints"""
        if component_idx is None:
            # Return summary for all components
            info = {}
            for comp_idx in range(self.pose_ncomp):
                info[comp_idx] = self.top_joints_per_component[comp_idx]
            return info
        elif 0 <= component_idx < self.pose_ncomp:
            # Return info for specific component
            return self.top_joints_per_component[component_idx]
        else:
            raise ValueError(f"Component index must be between 0 and {self.pose_ncomp-1}")
    
    def generate_hand_mesh(self, pose_param=None, shape_param=None, tex_param=None):
        """Generate a hand mesh using specified parameters"""
        # Use default parameters if not provided
        if pose_param is None:
            pose_param = torch.zeros(1, 30, device=self.device)
        if shape_param is None:
            shape_param = self.default_shape
        if tex_param is None:
            tex_param = self.default_texture
        
        # Ensure parameters are on the correct device and have batch dimension
        if pose_param.dim() == 1:
            pose_param = pose_param.unsqueeze(0)
        if shape_param.dim() == 1:
            shape_param = shape_param.unsqueeze(0)
        if tex_param is not None and tex_param.dim() == 1:
            tex_param = tex_param.unsqueeze(0)
        
        pose_param = pose_param.to(self.device)
        shape_param = shape_param.to(self.device)
        if tex_param is not None:
            tex_param = tex_param.to(self.device)
        
        # Generate hand mesh
        skin_v, muscle_v, bone_v, bone_joints, tex_img = self.nlayer.forward(
            pose_param, shape_param, tex_param, handle_collision=True
        )
        
        # Create and smooth mesh
        skin_p3dmesh = Meshes(skin_v, self.nlayer.skin_f.repeat(skin_v.shape[0], 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        
        return {
            'skin_mesh': skin_p3dmesh,
            'skin_v': skin_p3dmesh.verts_padded(),
            'skin_f': self.nlayer.skin_f,
            'muscle_v': muscle_v,
            'muscle_f': self.nlayer.muscle_f,
            'bone_v': bone_v,
            'bone_f': self.nlayer.bone_f,
            'joints': bone_joints,
            'texture': tex_img
        }
    
    def save_hand_mesh(self, mesh_data, output_path, save_texture=True):
        """Save the hand mesh to a file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save skin mesh
        pytorch3d.io.IO().save_mesh(mesh_data['skin_mesh'][0], output_path)
        
        # Save with texture if requested
        if save_texture and mesh_data['texture'] is not None:
            texture_path = output_path.replace('.obj', '_textured.obj')
            save_textured_nimble(
                texture_path,
                mesh_data['skin_v'][0].detach().cpu().numpy(),
                mesh_data['texture'][0].detach().cpu().numpy()
            )
        
        # Also save joints for reference
        joints_path = output_path.replace('.obj', '_joints.xyz')
        np.savetxt(joints_path, mesh_data['joints'][0].detach().cpu().numpy())
        
        print(f"Saved mesh to {output_path}")
    
    def test_component(self, component_idx, values=[-2.0, -1.0, 0.0, 1.0, 2.0], output_dir="component_tests"):
        """Test a specific pose component with different values"""
        os.makedirs(output_dir, exist_ok=True)
        
        for value in values:
            # Create pose parameter with only this component active
            pose_param = torch.zeros(1, 30, device=self.device)
            pose_param[0, component_idx] = value
            
            # Generate and save mesh
            mesh_data = self.generate_hand_mesh(pose_param)
            output_path = os.path.join(output_dir, f"component_{component_idx}_value_{value}.obj")
            self.save_hand_mesh(mesh_data, output_path)
            
            print(f"Generated mesh for component {component_idx}, value {value}")
    
    def generate_number_gesture(self, number, output_dir="number_gestures"):
        """Generate a specific number gesture (1-10)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Define gesture parameters for numbers 1-10
        gesture_params = {
            # These parameters need to be adjusted based on testing
            1: {0: 2.0, 4: -2.0, 8: -2.0, 12: -2.0, 16: -2.0},  # Index finger only
            2: {0: 2.0, 4: 2.0, 8: -2.0, 12: -2.0, 16: -2.0},   # Index and middle
            3: {0: 2.0, 4: 2.0, 8: 2.0, 12: -2.0, 16: -2.0},    # Index, middle, ring
            4: {0: 2.0, 4: 2.0, 8: 2.0, 12: 2.0, 16: -2.0},     # All except pinky
            5: {0: 2.0, 4: 2.0, 8: 2.0, 12: 2.0, 16: 2.0},      # All fingers
            6: {0: -2.0, 1: 2.0, 4: 2.0, 8: 2.0, 12: 2.0, 16: -2.0},  # Custom for 6
            7: {0: -2.0, 1: 2.0, 4: 2.0, 8: 2.0, 12: -2.0, 16: -2.0}, # Custom for 7
            8: {0: -2.0, 1: 2.0, 4: 2.0, 8: -2.0, 12: -2.0, 16: -2.0}, # Custom for 8
            9: {0: -2.0, 1: 2.0, 4: -2.0, 8: -2.0, 12: -2.0, 16: -2.0}, # Custom for 9
            10: {1: 2.0, 0: -2.0, 4: -2.0, 8: -2.0, 12: -2.0, 16: -2.0} # Thumb only for 10
        }
        
        if number not in gesture_params:
            raise ValueError(f"Number must be between 1 and 10, got {number}")
        
        # Create pose parameter tensor
        pose_param = torch.zeros(1, 30, device=self.device)
        for idx, value in gesture_params[number].items():
            pose_param[0, idx] = value
        
        # Generate and save mesh
        mesh_data = self.generate_hand_mesh(pose_param)
        output_path = os.path.join(output_dir, f"number_{number}.obj")
        self.save_hand_mesh(mesh_data, output_path)
        
        # Also save the parameters for reference
        param_path = os.path.join(output_dir, f"number_{number}_params.txt")
        with open(param_path, 'w') as f:
            f.write(f"Pose parameters for number {number}:\n")
            for i, val in enumerate(pose_param[0].detach().cpu().numpy()):
                f.write(f"Param {i}: {val:.4f}\n")
        
        print(f"Generated hand gesture for number {number}")
        return mesh_data
    
    def generate_all_number_gestures(self, output_dir="number_gestures"):
        """Generate hand gestures for all numbers 1-10"""
        for number in range(1, 11):
            self.generate_number_gesture(number, output_dir)
    
    def custom_pose(self, param_dict, output_path=None):
        """
        Generate a hand pose with custom parameter values.
        
        Args:
            param_dict: Dictionary mapping from parameter index to value
            output_path: Path to save the mesh, or None to not save
        
        Returns:
            Mesh data dictionary
        """
        pose_param = torch.zeros(1, 30, device=self.device)
        for idx, value in param_dict.items():
            pose_param[0, idx] = value
        
        mesh_data = self.generate_hand_mesh(pose_param)
        
        if output_path:
            self.save_hand_mesh(mesh_data, output_path)
        
        return mesh_data

def print_usage():
    print("\nNIMBLE Hand Pose Generator Usage:")
    print("=================================")
    print("This script helps you explore and generate hand poses using the NIMBLE model.")
    print("\nCommands:")
    print("  1. Test individual pose parameters:           python hand_pose_generator.py test-components")
    print("  2. Generate hand gestures for numbers 1-10:   python hand_pose_generator.py numbers")
    print("  3. Generate a specific number gesture:        python hand_pose_generator.py number <1-10>")
    print("  4. View information about pose components:    python hand_pose_generator.py component-info")
    print("  5. Test a specific component:                 python hand_pose_generator.py test-component <idx>")
    print("  6. Create a custom pose:                      python hand_pose_generator.py custom")
    print("\nExamples:")
    print("  python hand_pose_generator.py test-components")
    print("  python hand_pose_generator.py number 5")
    print("  python hand_pose_generator.py test-component 3")

if __name__ == "__main__":
    import sys
    
    # Check if we should print usage information
    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Initialize the hand pose generator
    generator = HandPoseGenerator()
    
    # Parse command line arguments
    command = sys.argv[1]
    
    if command == "test-components":
        print("Testing all pose components...")
        os.makedirs("component_tests", exist_ok=True)
        for i in range(30):
            generator.test_component(i, values=[-2.0, 0.0, 2.0])
            # Clean up GPU memory
            torch.cuda.empty_cache()
        print("Component testing complete!")
    
    elif command == "numbers":
        print("Generating hand gestures for numbers 1-10...")
        generator.generate_all_number_gestures()
        print("All number gestures generated!")
    
    elif command == "number":
        if len(sys.argv) < 3:
            print("Error: Please specify which number (1-10)")
            print_usage()
            sys.exit(1)
        number = int(sys.argv[2])
        if number < 1 or number > 10:
            print(f"Error: Number must be between 1 and 10, got {number}")
            sys.exit(1)
        print(f"Generating hand gesture for number {number}...")
        generator.generate_number_gesture(number)
    
    elif command == "component-info":
        info = generator.get_component_info()
        print("\nComponent to Joint Effect Analysis:")
        print("=================================")
        for comp_idx, data in info.items():
            joints_str = ", ".join([f"Joint {j} ({data['effects'][i]:.3f})" 
                                   for i, j in enumerate(data['joints'])])
            print(f"Component {comp_idx}: Top joints = {joints_str}")
    
    elif command == "test-component":
        if len(sys.argv) < 3:
            print("Error: Please specify which component to test (0-29)")
            print_usage()
            sys.exit(1)
        component_idx = int(sys.argv[2])
        if component_idx < 0 or component_idx >= 30:
            print(f"Error: Component index must be between 0 and 29, got {component_idx}")
            sys.exit(1)
        print(f"Testing pose component {component_idx}...")
        generator.test_component(component_idx)
    
    elif command == "custom":
        print("Creating a custom hand pose...")
        print("Enter parameter values in format 'index value', e.g., '0 2.0'")
        print("Enter 'done' when finished.")
        
        param_dict = {}
        while True:
            inp = input("> ")
            if inp.lower() == 'done':
                break
            try:
                idx, val = inp.split()
                idx, val = int(idx), float(val)
                if 0 <= idx < 30:
                    param_dict[idx] = val
                    print(f"Set parameter {idx} to {val}")
                else:
                    print(f"Error: Index must be between 0 and 29, got {idx}")
            except ValueError:
                print("Error: Input must be in format 'index value'")
        
        if param_dict:
            output_path = "custom_pose.obj"
            generator.custom_pose(param_dict, output_path)
            print(f"Custom pose saved to {output_path}")
        else:
            print("No parameters specified, aborting.")
    
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
