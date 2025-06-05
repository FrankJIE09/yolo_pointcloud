import sys
import os
import datetime
import copy
import numpy as np
import argparse
import time
import json # For loading args
import glob # For finding instance files
import yaml # Added for YAML support
import torch # Added for PyTorch ICP

# --- 导入 Open3D ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install Open3D (pip install open3d).")
    sys.exit(1)

# --- PyTorch ICP Helper Functions ---
def find_nearest_neighbors_torch(source_points, target_points):
    # source_points: (N, 3) tensor
    # target_points: (M, 3) tensor
    # Returns: indices of target_points nearest to each source_point (N,), and squared distances (N,)
    dist_sq = torch.cdist(source_points, target_points).pow(2)  # (N, M) squared Euclidean distances
    min_dist_sq, indices = torch.min(dist_sq, dim=1)
    return indices, min_dist_sq

def estimate_point_to_point_svd_torch(P, Q): # P=source, Q=target, (K,3) tensors
    centroid_P = torch.mean(P, dim=0, keepdim=True)
    centroid_Q = torch.mean(Q, dim=0, keepdim=True)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered # Covariance matrix (3, 3)
    
    try:
        U, S, Vt = torch.linalg.svd(H)
    except torch.linalg.LinAlgError as e:
        print(f"SVD failed: {e}. Returning identity transform.")
        T = torch.eye(4, device=P.device, dtype=P.dtype)
        return T
        
    R = Vt.T @ U.T
    # Ensure a right-handed coordinate system
    if torch.linalg.det(R) < 0:
        Vt_copy = Vt.clone()
        Vt_copy[2, :] *= -1
        R = Vt_copy.T @ U.T
        
    t = centroid_Q.T - R @ centroid_P.T
    
    T = torch.eye(4, device=P.device, dtype=P.dtype)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    return T

def estimate_point_to_plane_least_squares_torch(P_transformed, Q_corr, Q_normals_corr, device):
    # P_transformed: Transformed source points in correspondence (K, 3)
    # Q_corr: Corresponding target points (K, 3)
    # Q_normals_corr: Normals of corresponding target points (K, 3)
    # Returns: delta_transform (4x4 tensor) to refine P_transformed towards Q_corr
    
    K = P_transformed.shape[0]
    A = torch.zeros(K, 6, device=device, dtype=P_transformed.dtype)
    b = torch.zeros(K, device=device, dtype=P_transformed.dtype)

    # For linearized rotation R_delta = I + [0 -gamma beta; gamma 0 -alpha; -beta alpha 0]
    # Error: ((R_delta @ P_i + t_delta) - Q_i) . n_i = 0
    # Approx: (P_i + omega x P_i + t_delta - Q_i) . n_i = 0
    # ( (omega x P_i) + t_delta ) . n_i = (Q_i - P_i) . n_i
    # omega . (P_i x n_i) + t_delta . n_i = (Q_i - P_i) . n_i
    
    # P_i x n_i
    cross_prod_P_n = torch.cross(P_transformed, Q_normals_corr, dim=1)
    
    A[:, 0] = cross_prod_P_n[:, 0] # for alpha (rotation about x-axis)
    A[:, 1] = cross_prod_P_n[:, 1] # for beta (rotation about y-axis)
    A[:, 2] = cross_prod_P_n[:, 2] # for gamma (rotation about z-axis)
    
    A[:, 3] = Q_normals_corr[:, 0] # for tx
    A[:, 4] = Q_normals_corr[:, 1] # for ty
    A[:, 5] = Q_normals_corr[:, 2] # for tz
    
    b = torch.sum((Q_corr - P_transformed) * Q_normals_corr, dim=1)

    try:
        # Solve Ax = b for x = [alpha, beta, gamma, tx, ty, tz]
        x = torch.linalg.lstsq(A, b).solution
    except torch.linalg.LinAlgError as e:
        print(f"Point-to-plane lstsq failed: {e}. Returning identity delta transform.")
        return torch.eye(4, device=device, dtype=P_transformed.dtype)

    alpha, beta, gamma = x[0], x[1], x[2]
    tx, ty, tz = x[3], x[4], x[5]

    # Create delta rotation matrix from small angles
    # R_delta approx I + skew(omega)
    R_delta = torch.eye(3, device=device, dtype=P_transformed.dtype)
    R_delta[0, 1] = -gamma
    R_delta[0, 2] = beta
    R_delta[1, 0] = gamma
    R_delta[1, 2] = -alpha
    R_delta[2, 0] = -beta
    R_delta[2, 1] = alpha
    
    # More robust way to get R_delta from omega_vec = [alpha, beta, gamma]
    omega_vec = x[:3]
    angle = torch.norm(omega_vec)
    if angle > 1e-9: # Avoid division by zero if angle is very small
        axis = omega_vec / angle
        K_axis = torch.zeros((3, 3), device=device, dtype=P_transformed.dtype)
        K_axis[0, 1] = -axis[2]
        K_axis[0, 2] = axis[1]
        K_axis[1, 0] = axis[2]
        K_axis[1, 2] = -axis[0]
        K_axis[2, 0] = -axis[1]
        K_axis[2, 1] = axis[0]
        R_delta = torch.eye(3, device=device, dtype=P_transformed.dtype) + \
                  torch.sin(angle) * K_axis + \
                  (1 - torch.cos(angle)) * (K_axis @ K_axis)

    delta_transform = torch.eye(4, device=device, dtype=P_transformed.dtype)
    delta_transform[:3, :3] = R_delta
    delta_transform[:3, 3] = x[3:] # tx, ty, tz
    
    return delta_transform

def pytorch_icp_registration(
    source_points_tensor, # Original source points (N, 3)
    target_points_tensor, # Target points (M, 3)
    initial_transform_guess, # Initial transform (4, 4)
    max_iterations,
    distance_threshold, # For correspondence filtering
    rmse_change_threshold, # Absolute change in RMSE for convergence
    transform_change_threshold, # Norm of change in transform for convergence
    estimation_method='point_to_point',
    target_normals_tensor=None, # Required for point-to-plane (M,3)
    device=torch.device("cpu")
):
    if estimation_method == 'point_to_plane' and target_normals_tensor is None:
        raise ValueError("Target normals must be provided for point-to-plane ICP.")

    current_transform = initial_transform_guess.clone().to(device=device, dtype=source_points_tensor.dtype)
    source_homo = torch.cat([source_points_tensor, torch.ones(source_points_tensor.shape[0], 1, device=device, dtype=source_points_tensor.dtype)], dim=1).T

    prev_rmse = float('inf')
    
    for i in range(max_iterations):
        transformed_source_points = (current_transform @ source_homo)[:3, :].T

        # Find correspondences
        corr_indices_target, dist_sq = find_nearest_neighbors_torch(transformed_source_points, target_points_tensor)
        
        # Filter correspondences based on distance_threshold
        valid_mask = dist_sq < (distance_threshold**2)
        num_correspondences = valid_mask.sum().item()

        if num_correspondences < 10: # Min correspondences to proceed
            print(f"Iter {i+1}/{max_iterations}: Too few correspondences ({num_correspondences}). Stopping.")
            break

        P_corr_transformed = transformed_source_points[valid_mask]
        Q_corr_target = target_points_tensor[corr_indices_target[valid_mask]]
        
        current_rmse = torch.sqrt(torch.mean(dist_sq[valid_mask])).item() # RMSE of inlier distances
        fitness = num_correspondences / source_points_tensor.shape[0]

        transform_update_matrix = torch.eye(4, device=device, dtype=source_points_tensor.dtype)
        if estimation_method == 'point_to_point':
            # Estimate transform from original source points in correspondence to target points
            P_orig_corr = source_points_tensor[valid_mask]
            new_total_transform = estimate_point_to_point_svd_torch(P_orig_corr, Q_corr_target)
            transform_update_matrix = new_total_transform @ torch.linalg.inv(current_transform) # For measuring change
            current_transform = new_total_transform
        elif estimation_method == 'point_to_plane':
            Q_normals_corr = target_normals_tensor[corr_indices_target[valid_mask]]
            # Delta transform refines current_transform: new_transform = delta @ current_transform
            transform_update_matrix = estimate_point_to_plane_least_squares_torch(P_corr_transformed, Q_corr_target, Q_normals_corr, device)
            current_transform = transform_update_matrix @ current_transform
        
        # Check for convergence
        delta_transform_norm = torch.norm(transform_update_matrix - torch.eye(4, device=device, dtype=source_points_tensor.dtype)).item()
        rmse_diff = abs(prev_rmse - current_rmse)

        print(f"Iter {i+1}/{max_iterations}: RMSE: {current_rmse:.6f}, Fitness: {fitness:.6f}, Corr: {num_correspondences}, RMSE_Change: {rmse_diff:.6g}, dT_Norm: {delta_transform_norm:.6g}")

        if i > 0: # Allow at least one iteration
            if rmse_diff < rmse_change_threshold:
                print(f"Converged at iter {i+1} due to RMSE change tolerance.")
                break
            if delta_transform_norm < transform_change_threshold:
                print(f"Converged at iter {i+1} due to transform change tolerance.")
                break
        
        prev_rmse = current_rmse
        if i == max_iterations -1:
            print("Reached max iterations.")

    # Final metrics based on the final transform
    final_transformed_source = (current_transform @ source_homo)[:3, :].T
    final_corr_indices, final_dist_sq = find_nearest_neighbors_torch(final_transformed_source, target_points_tensor)
    final_valid_mask = final_dist_sq < (distance_threshold**2)
    
    final_num_correspondences = final_valid_mask.sum().item()
    final_fitness = final_num_correspondences / source_points_tensor.shape[0]
    if final_num_correspondences > 0:
        final_inlier_rmse = torch.sqrt(torch.mean(final_dist_sq[final_valid_mask])).item()
    else:
        final_inlier_rmse = float('inf')

    return {
        "transformation": current_transform.cpu().numpy(),
        "fitness": final_fitness,
        "inlier_rmse": final_inlier_rmse,
        "correspondence_set_size": final_num_correspondences 
    }

# --- 可视化函数 (从 Part 1 复制过来，确保一致性) ---
def visualize_single_pcd(pcd, window_name="Point Cloud", point_color=[0.5, 0.5, 0.5]):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if not pcd.has_points(): print(f"Skipping visualization for {window_name}: No points."); return
    pcd_vis = copy.deepcopy(pcd)
    if not pcd_vis.has_colors(): pcd_vis.paint_uniform_color(point_color)
    print(f"\nDisplaying Point Cloud: {window_name} (Points: {len(pcd_vis.points)})")
    print("(Close the Open3D window to continue script execution...)")
    try:
        o3d.visualization.draw_geometries([pcd_vis], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")

def visualize_icp_alignment(source_pcd_transformed_model, instance_pcd_observed, window_name="ICP Alignment Result"):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    # Ensure inputs are Open3D point clouds for visualization
    
    # Convert source_pcd_transformed_model if it's a NumPy array or tensor
    if isinstance(source_pcd_transformed_model, (np.ndarray, torch.Tensor)):
        temp_pcd = o3d.geometry.PointCloud()
        if isinstance(source_pcd_transformed_model, torch.Tensor):
            source_pcd_transformed_model = source_pcd_transformed_model.cpu().numpy()
        temp_pcd.points = o3d.utility.Vector3dVector(source_pcd_transformed_model)
        source_pcd_transformed_model = temp_pcd
        
    # Convert instance_pcd_observed if it's a NumPy array or tensor
    if isinstance(instance_pcd_observed, (np.ndarray, torch.Tensor)):
        temp_pcd = o3d.geometry.PointCloud()
        if isinstance(instance_pcd_observed, torch.Tensor):
            instance_pcd_observed = instance_pcd_observed.cpu().numpy()
        temp_pcd.points = o3d.utility.Vector3dVector(instance_pcd_observed)
        instance_pcd_observed = temp_pcd

    if source_pcd_transformed_model.has_points() or instance_pcd_observed.has_points():
        instance_pcd_vis = copy.deepcopy(instance_pcd_observed)
        transformed_model_vis = copy.deepcopy(source_pcd_transformed_model)
        
        if not transformed_model_vis.has_colors(): transformed_model_vis.paint_uniform_color([1, 0.706, 0])  # Yellow
        if not instance_pcd_vis.has_colors(): instance_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue
            
        print(f"\nDisplaying ICP Alignment: {window_name}...")
        print("Yellow: Transformed Model Sampled Points | Blue: Observed Instance Points (preprocessed)")
        print("(Close the Open3D window to continue script execution...)")
        try:
            o3d.visualization.draw_geometries([transformed_model_vis, instance_pcd_vis], window_name=window_name)
            print(f"Alignment visualization window '{window_name}' closed.")
        except Exception as e:
            print(f"Error during visualization: {e}")

def visualize_transformed_model_in_scene(scene_pcd, target_model_geometry, transform, window_name="Transformed Model in Scene"):
    """
    Visualize the transformed model in the original scene.
    模板始终显示为纯色（如红色），不带纹理，便于区分。
    """
    # Create a copy of the scene point cloud to avoid modifying the original
    scene_vis = o3d.geometry.PointCloud(scene_pcd)
    
    # Create a copy of the target model geometry
    if isinstance(target_model_geometry, o3d.geometry.TriangleMesh):
        model_vis = o3d.geometry.TriangleMesh(target_model_geometry)
    else:
        model_vis = o3d.geometry.PointCloud(target_model_geometry)
    
    # Apply the transformation to the model
    model_vis.transform(transform)
    
    # Set colors for visualization
    if not scene_vis.has_colors(): 
        print("警告：原始场景点云没有颜色信息")
        scene_vis.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
    else:
        print("使用原始场景点云的颜色信息")

    # 模板始终显示为红色，不带纹理
    if isinstance(model_vis, o3d.geometry.TriangleMesh):
        print("模板以红色显示（无纹理）")
        model_vis.paint_uniform_color([1, 0, 0])  # 红色
    else:
        print("模板以红色显示（无纹理）")
        model_vis.paint_uniform_color([1, 0, 0])  # 红色
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # Add geometries to the visualizer
    vis.add_geometry(scene_vis)
    vis.add_geometry(model_vis)
    
    # Set up the view
    vis.get_render_option().point_size = 2
    vis.get_render_option().background_color = np.array([0, 0, 0])
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Namespace class to convert dict to object for args
class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

# --- Helper function to load config from YAML (copied from Part 1) ---
def load_config_from_yaml(config_path):
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Successfully loaded configuration from {config_path}")
            return config_data
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {config_path}: {e}. Using default script arguments.")
            return {}
        except Exception as e:
            print(f"Warning: Could not read YAML file {config_path}: {e}. Using default script arguments.")
            return {}
    else:
        return {}

# --- Helper function to get value from config dict or use default (copied from Part 1) ---
def get_config_value(config_dict, section_name, key_name, default_value):
    if config_dict and section_name in config_dict and key_name in config_dict[section_name]:
        val = config_dict[section_name][key_name]
        # Ensure None from YAML is treated as Python None for BooleanOptionalAction defaults
        if isinstance(default_value, type(None)) and isinstance(val, str) and val.lower() == 'null':
            return None
        return val
    return default_value

def main(cli_args_part2):
    print(f"Starting Part 2 (ICP Estimation from Intermediate Data) at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Loading intermediate data from: {cli_args_part2.intermediate_dir}")

    if not os.path.isdir(cli_args_part2.intermediate_dir):
        print(f"FATAL Error: Intermediate directory not found: {cli_args_part2.intermediate_dir}")
        sys.exit(1)

    # Load original args from part 1 (contains args.no_cuda)
    args_file_path = os.path.join(cli_args_part2.intermediate_dir, "args.json")
    if not os.path.exists(args_file_path):
        print(f"FATAL Error: args.json not found in {cli_args_part2.intermediate_dir}")
        sys.exit(1)
    try:
        with open(args_file_path, 'r') as f:
            args_from_part1_dict = json.load(f)
        args_from_part1 = ArgsNamespace(**args_from_part1_dict) 
        print("Successfully loaded original arguments from args.json (from Part 1)")
        # print(f"Original arguments (from Part 1): {args_from_part1}") # Can be verbose
    except Exception as e:
        print(f"FATAL Error loading or parsing args.json: {e}")
        sys.exit(1)

    # Determine PyTorch device
    use_cuda_from_part1 = not getattr(args_from_part1, 'no_cuda', True) # Default to no_cuda if not present
    if use_cuda_from_part1 and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch ICP will use CUDA device as per Part 1 configuration (no_cuda={args_from_part1.no_cuda}).")
    else:
        if use_cuda_from_part1 and not torch.cuda.is_available():
            print("Warning: CUDA requested by Part 1 config but not available. PyTorch ICP will use CPU.")
        device = torch.device("cpu")
        print("PyTorch ICP will use CPU.")
    
    # Part 2 CLI args (cli_args_part2) override visualization, output, and now ICP params from YAML/defaults
    # The 'args' object used for visualization flags needs to be managed carefully.
    # Let's create a combined args object where cli_args_part2 override args_from_part1 for shared keys
    # For visualization and output dir, Part 2's CLI/YAML settings should take precedence.
    
    effective_args = copy.deepcopy(args_from_part1) # Start with args from part1 (args.json)
    
    # Ensure critical Part 2 flags exist on effective_args, defaulting if not from args.json
    if not hasattr(effective_args, 'visualize_pose'):
        effective_args.visualize_pose = False
    if not hasattr(effective_args, 'visualize_pose_in_scene'):
        effective_args.visualize_pose_in_scene = False
    if not hasattr(effective_args, 'save_results'):
        effective_args.save_results = False
    
    # Override with Part 2's specific CLI/YAML settings if they are provided (not None)
    if cli_args_part2.visualize_pose is not None:
        effective_args.visualize_pose = cli_args_part2.visualize_pose
        
    if cli_args_part2.visualize_pose_in_scene is not None:
        effective_args.visualize_pose_in_scene = cli_args_part2.visualize_pose_in_scene
        
    if cli_args_part2.save_results is not None:
        effective_args.save_results = cli_args_part2.save_results
    
    if cli_args_part2.output_dir_part2 is not None: 
        effective_args.output_dir = cli_args_part2.output_dir_part2 
    elif hasattr(effective_args, 'output_dir') and effective_args.output_dir is not None:
        # If Part 2 output_dir is not specified, but Part 1 output_dir exists,
        # create a subfolder in Part 1's output_dir for Part 2 results
        effective_args.output_dir = os.path.join(effective_args.output_dir, "part2_results_pytorch_icp")
    else: # Fallback if Part 1 output_dir was also None (should not happen with defaults)
        effective_args.output_dir = os.path.join(cli_args_part2.intermediate_dir, "part2_results_pytorch_icp")
        
    os.makedirs(effective_args.output_dir, exist_ok=True)
    print(f"Final poses will be saved to: {effective_args.output_dir}")
    print(f"Using effective_args for visualization and output: {vars(effective_args)}")


    # --- Load Common Intermediate Data (Open3D objects) ---
    path_common_target_centered_o3d = os.path.join(cli_args_part2.intermediate_dir, "common_target_model_centered.pcd")
    path_common_target_original_scale_o3d = os.path.join(cli_args_part2.intermediate_dir, "common_target_model_original_scale.pcd")
    path_common_target_centroid_np = os.path.join(cli_args_part2.intermediate_dir, "common_target_centroid_original_model_scale.npy")
    path_common_original_scene = os.path.join(cli_args_part2.intermediate_dir, "common_original_scene.pcd")
    path_model_file_txt = os.path.join(cli_args_part2.intermediate_dir, "model_file_path.txt")

    try:
        print("Loading common intermediate data...")
        target_pcd_model_centered_o3d = o3d.io.read_point_cloud(path_common_target_centered_o3d)
        target_pcd_original_model_scale_o3d = o3d.io.read_point_cloud(path_common_target_original_scale_o3d)
        target_centroid_original_model_scale_np = np.load(path_common_target_centroid_np)
        original_scene_pcd = None
        if os.path.exists(path_common_original_scene):
            try:
                original_scene_pcd = o3d.io.read_point_cloud(path_common_original_scene)
                print(f"DEBUG (Part 2): Loaded original_scene_pcd from {path_common_original_scene}")
                if not original_scene_pcd.has_points():
                    print("  WARNING (Part 2): Loaded original_scene_pcd has no points.")
                    original_scene_pcd = None # Treat as not loaded if empty
                else:
                    print(f"  DEBUG (Part 2): original_scene_pcd has {len(original_scene_pcd.points)} points.")
                    print(f"  DEBUG (Part 2): original_scene_pcd.has_colors(): {original_scene_pcd.has_colors()}")
                    if original_scene_pcd.has_colors():
                        print(f"  DEBUG (Part 2): First 3 color values of original_scene_pcd: {np.asarray(original_scene_pcd.colors)[:3]}")
                    else:
                        print("  DEBUG (Part 2): original_scene_pcd loaded WITHOUT colors.")
            except Exception as e_load_scene:
                print(f"  WARNING (Part 2): Could not load common_original_scene.pcd: {e_load_scene}")
                original_scene_pcd = None
        else:
            print(f"  WARNING (Part 2): common_original_scene.pcd not found at {path_common_original_scene}")
        
        with open(path_model_file_txt, 'r') as f_model_path: loaded_model_file_path = f_model_path.read().strip()
        target_mesh_for_scene_viz = o3d.geometry.TriangleMesh() 
        if os.path.exists(loaded_model_file_path):
            try:
                temp_mesh_viz = o3d.io.read_triangle_mesh(loaded_model_file_path)
                if temp_mesh_viz.has_vertices(): target_mesh_for_scene_viz = temp_mesh_viz
            except Exception: pass # Ignore if not a mesh
        if not target_mesh_for_scene_viz.has_vertices(): # Fallback to point cloud if mesh loading failed or not a mesh
             target_mesh_for_scene_viz = target_pcd_original_model_scale_o3d


        if not target_pcd_model_centered_o3d.has_points() or \
           not target_pcd_original_model_scale_o3d.has_points() or \
           not original_scene_pcd.has_points():
            raise ValueError("One or more common PCD files failed to load or are empty.")
        print("Common data loaded successfully.")
    except Exception as e:
        print(f"FATAL Error loading common intermediate data: {e}")
        sys.exit(1)

    # --- Find and Process Instance Data ---
    instance_preprocessed_files = sorted(glob.glob(os.path.join(cli_args_part2.intermediate_dir, "instance_*_preprocessed.pcd")))
    if not instance_preprocessed_files:
        print("No preprocessed instance files found in the intermediate directory. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(instance_preprocessed_files)} instances to process.")
    estimated_poses = {}

    # Convert common target model (centered) to tensor once
    target_points_centered_np = np.asarray(target_pcd_model_centered_o3d.points, dtype=np.float32)
    target_points_centered_tensor = torch.from_numpy(target_points_centered_np).float().to(device) # Ensure float32
    target_normals_centered_tensor = None

    if cli_args_part2.icp_estimation_method.lower() == 'point_to_plane':
        print(f"Estimating normals for target model for Point-to-Plane ICP...")
        normal_radius_for_target_icp = cli_args_part2.icp_threshold * 2.0 # Heuristic for normal radius
        normal_radius_for_target_icp = max(1e-3, normal_radius_for_target_icp) # Ensure positive radius
        target_pcd_model_centered_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_for_target_icp, max_nn=30))
        if target_pcd_model_centered_o3d.has_normals():
            target_normals_centered_np = np.asarray(target_pcd_model_centered_o3d.normals, dtype=np.float32)
            target_normals_centered_tensor = torch.from_numpy(target_normals_centered_np).float().to(device) # Ensure float32
            print(f"  Target normals estimated and converted to tensor (shape: {target_normals_centered_tensor.shape}).")
        else:
            print("  Warning: Failed to estimate normals for target model. Point-to-Plane ICP might be suboptimal or fail.")
            # Consider falling back to point_to_point if normals fail critically
            # cli_args_part2.icp_estimation_method = 'point_to_point'
            # print("  Falling back to point_to_point estimation method.")


    for inst_pcd_file in instance_preprocessed_files:
        try:
            basename = os.path.basename(inst_pcd_file)
            parts = basename.split('_')
            if len(parts) < 3 or parts[0] != 'instance' or parts[-1] != 'preprocessed.pcd':
                print(f"Skipping unrecognized file format: {basename}")
                continue
            inst_id_str = parts[1]
            try: inst_id = int(inst_id_str)
            except ValueError: print(f"Skipping file with non-integer instance ID: {basename}"); continue
            
            print(f"\nProcessing Instance ID: {inst_id}")

            path_inst_centroid_np = os.path.join(cli_args_part2.intermediate_dir, f"instance_{inst_id_str}_centroid.npy")
            path_inst_pca_transform_np = os.path.join(cli_args_part2.intermediate_dir, f"instance_{inst_id_str}_pca_transform.npy")

            if not os.path.exists(path_inst_centroid_np) or not os.path.exists(path_inst_pca_transform_np):
                print(f"  Missing centroid or PCA transform file for instance {inst_id}. Skipping.")
                continue

            instance_pcd_preprocessed_o3d = o3d.io.read_point_cloud(inst_pcd_file)
            instance_centroid_for_icp_np = np.load(path_inst_centroid_np)
            initial_transform_icp_np = np.load(path_inst_pca_transform_np)

            if not instance_pcd_preprocessed_o3d.has_points():
                print(f"  Instance {inst_id} preprocessed PCD is empty. Skipping.")
                continue
            
            instance_points_preprocessed_np = np.asarray(instance_pcd_preprocessed_o3d.points, dtype=np.float32)
            
            # Center the instance points (as they were for PCA in Part 1) for ICP input
            instance_points_centered_np = instance_points_preprocessed_np - instance_centroid_for_icp_np
            
            print(f"  Loaded preprocessed instance data. Centered instance points: {instance_points_centered_np.shape[0]}")

            if instance_points_centered_np.shape[0] < cli_args_part2.icp_min_points:
                print(f"  Skipping Inst {inst_id}: Not enough points for ICP ({instance_points_centered_np.shape[0]} < {cli_args_part2.icp_min_points}).")
                continue

            # Convert to PyTorch tensors
            source_points_tensor = torch.from_numpy(instance_points_centered_np).float().to(device) # Ensure float32
            # Ensure initial_transform_tensor matches the dtype of source_points_tensor
            initial_transform_tensor = torch.from_numpy(initial_transform_icp_np).to(device=device, dtype=source_points_tensor.dtype)
            
            print(f"    Running PyTorch ICP (Method: {cli_args_part2.icp_estimation_method}, Device: {device})...")
            start_time_icp = time.perf_counter()
            
            icp_result_dict = pytorch_icp_registration(
                source_points_tensor=source_points_tensor,
                target_points_tensor=target_points_centered_tensor,
                initial_transform_guess=initial_transform_tensor,
                max_iterations=cli_args_part2.icp_max_iter,
                distance_threshold=cli_args_part2.icp_threshold,
                rmse_change_threshold=cli_args_part2.icp_relative_rmse, # Using relative_rmse as absolute RMSE change threshold
                transform_change_threshold=1e-5, # Default transform change threshold
                estimation_method=cli_args_part2.icp_estimation_method.lower(),
                target_normals_tensor=target_normals_centered_tensor,
                device=device
            )
            
            end_time_icp = time.perf_counter()
            duration_icp = end_time_icp - start_time_icp

            # ICP result is a NumPy array (transformation)
            T_centered_s_to_centered_t_np = icp_result_dict["transformation"]
            
            # Final pose composition (instance centroid -> world) @ (centered_instance -> centered_model) @ (model_centroid -> world)^-1
            # The PCA initial transform and ICP refinement BOTH operate on *centered* source and *centered* target.
            # So, T_centered_s_to_centered_t_np is the transformation from centered source to centered target.
            
            # T_source_original_to_world = T_translate_to_source_centroid @ T_centered_s_to_centered_t_np @ T_translate_to_target_origin_from_its_centroid @ T_target_original_to_world
            # We want to transform the *original scale CAD model* (whose origin might not be 0,0,0) to align with the *original scale instance*.
            # Original CAD model points: X_cad_orig
            # Centered CAD model points: X_cad_centered = X_cad_orig - centroid_cad_orig
            # Original instance points: X_inst_orig
            # Centered instance points: X_inst_centered = X_inst_orig - centroid_inst_orig
            
            # PCA gave: initial_transform_icp_np (maps centered_inst to centered_cad)
            # PyTorch ICP gave: T_centered_s_to_centered_t_np (maps centered_inst to centered_cad, refining initial_transform_icp_np)
            # This T_centered_s_to_centered_t_np is the *total* transform from centered source to centered target.
            
            # Transformation to apply to the *original target model* (target_pcd_original_model_scale_o3d)
            # to align it with the *original instance in the scene*.
            # 1. Translate original target model to its own centroid: T_to_target_centroid (-target_centroid_original_model_scale_np)
            # 2. Apply the computed transformation T_centered_s_to_centered_t_np
            # 3. Translate from new (centered) origin back to the instance's original world position: T_from_centered_to_instance_world (+instance_centroid_for_icp_np)
            
            T_target_orig_to_centered = np.eye(4)
            T_target_orig_to_centered[:3, 3] = -target_centroid_original_model_scale_np
            
            T_centered_target_to_instance_world = np.eye(4)
            T_centered_target_to_instance_world[:3, 3] = instance_centroid_for_icp_np
            
            # final_estimated_pose = T_centered_target_to_instance_world @ T_centered_s_to_centered_t_np @ T_target_orig_to_centered
            # This pose transforms the original CAD model (at its own origin) to match the observed instance.

            # Let P_cad be points of the CAD model in its original coord system.
            # Let P_inst be points of the instance in the scene's coord system.
            # We have: P_cad_centered = P_cad - C_cad
            #          P_inst_centered = P_inst - C_inst
            # T_final_centered transforms P_inst_centered to P_cad_centered.
            # So, P_cad_centered_aligned = T_final_centered @ P_inst_centered_homo
            # (P_cad - C_cad) = T_final_centered @ (P_inst - C_inst)_homo
            # We want T_pose such that T_pose @ P_cad_homo ~ P_inst_homo
            # P_inst ~ C_inst + T_final_centered^-1 @ (P_cad - C_cad)
            # T_pose should take P_cad to P_inst.
            # P_inst_world = Translation(C_inst) @ T_final_centered^-1 @ Translation(-C_cad) @ P_cad_world

            inv_T_centered = np.linalg.inv(T_centered_s_to_centered_t_np) # Maps centered_target to centered_source
            
            Translate_to_C_inst = np.eye(4)
            Translate_to_C_inst[:3,3] = instance_centroid_for_icp_np

            Translate_from_C_cad = np.eye(4)
            Translate_from_C_cad[:3,3] = -target_centroid_original_model_scale_np
            
            final_estimated_pose = Translate_to_C_inst @ inv_T_centered @ Translate_from_C_cad
            # This pose, when applied to the original CAD model, should align it with the instance in the scene.

            estimated_poses[f"instance_{inst_id_str}"] = final_estimated_pose 

            print(f"  PyTorch ICP for Inst {inst_id}:")
            print(f"    ICP Duration    : {duration_icp:.4f} seconds")
            print(f"    Fitness         : {icp_result_dict['fitness']:.6f}")
            print(f"    Inlier RMSE     : {icp_result_dict['inlier_rmse']:.6f}")
            print(f"    Correspondence Set: {icp_result_dict['correspondence_set_size']} pairs")
            # print(f"    Transformation (centered_instance to centered_model):\n{T_centered_s_to_centered_t_np}")


            if effective_args.visualize_pose:
                # For ICP alignment viz, show:
                # 1. The *centered instance* points (source_points_tensor)
                # 2. The *centered target* points (target_points_centered_tensor) transformed by T_centered_s_to_centered_t_np^-1
                # OR:
                # 1. The *centered instance* points transformed by T_centered_s_to_centered_t_np
                # 2. The *centered target* points
                
                # Let's show: transformed centered source vs centered target
                T_centered_s_to_centered_t_tensor = torch.from_numpy(T_centered_s_to_centered_t_np).to(device=device, dtype=source_points_tensor.dtype)
                viz_source_transformed_np = (T_centered_s_to_centered_t_tensor @ torch.cat([source_points_tensor, torch.ones(source_points_tensor.shape[0], 1, device=device, dtype=source_points_tensor.dtype)], dim=1).T)[:3, :].T.cpu().numpy()
                viz_target_centered_np = target_points_centered_tensor.cpu().numpy()
                
                # Create Open3D PCDs for visualization
                source_viz_pcd = o3d.geometry.PointCloud()
                source_viz_pcd.points = o3d.utility.Vector3dVector(viz_source_transformed_np)
                source_viz_pcd.paint_uniform_color([1, 0.706, 0]) # Yellow (transformed source)

                target_viz_pcd = o3d.geometry.PointCloud()
                target_viz_pcd.points = o3d.utility.Vector3dVector(viz_target_centered_np)
                target_viz_pcd.paint_uniform_color([0, 0.651, 0.929]) # Blue (target)
                
                print(f"\nDisplaying PyTorch ICP Alignment (Centered): Inst {inst_id}...")
                print("Yellow: Transformed Centered Instance | Blue: Centered Target Model")
                try:
                    o3d.visualization.draw_geometries([source_viz_pcd, target_viz_pcd], window_name=f"PyTorch ICP Align (Centered) - Inst {inst_id}")
                except Exception as e_vis: print(f"Error visualizing centered ICP: {e_vis}")

            
            if effective_args.visualize_pose_in_scene:
                # Use the original target geometry (mesh or full PCD) and apply final_estimated_pose
                # target_mesh_for_scene_viz is already loaded (original CAD model)
                visualize_transformed_model_in_scene(original_scene_pcd, 
                                                     target_mesh_for_scene_viz, # This is the original CAD model
                                                     final_estimated_pose,
                                                     window_name=f"Final CAD in Scene (PyTorch ICP) - Inst {inst_id}")
        except Exception as e_icp_inst:
            print(f"Error during PyTorch ICP processing for instance {inst_id}: {e_icp_inst}")
            import traceback; traceback.print_exc()

    if effective_args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        base_name_for_output = "unknown_input"
        if hasattr(args_from_part1, 'input_point_cloud_file') and args_from_part1.input_point_cloud_file is not None:
            base_name_for_output = os.path.splitext(os.path.basename(args_from_part1.input_point_cloud_file))[0]
        elif hasattr(args_from_part1, 'sample_index'):
            base_name_for_output = f"h5sample_{args_from_part1.sample_index}"
        
        pose_filename = os.path.join(effective_args.output_dir, f"estimated_poses_pytorch_{base_name_for_output}_{timestamp}.npz")
        try:
            np.savez(pose_filename, **estimated_poses)
            print(f"\nSaved PyTorch ICP poses to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving PyTorch ICP poses: {e_save}")
    
    print(f"\nPart 2 (PyTorch ICP Estimation) finished at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_file', type=str, default='pose_estimation_config.yaml', help='Path to the YAML configuration file.')
    cli_args_cfg, remaining_argv = pre_parser.parse_known_args() # Renamed cli_args to cli_args_cfg
    config_data = load_config_from_yaml(cli_args_cfg.config_file) 

    parser = argparse.ArgumentParser(description='Part 2: PyTorch ICP Pose Estimation from Intermediate Data.', parents=[pre_parser])
    parser.add_argument('--intermediate_dir', type=str, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'intermediate_dir', './pose_estimation_results_cmdline/intermediate_data_h5sample_0'),
                        help='Directory containing intermediate data from Part 1.')
    
    # Visualization and Output args for Part 2
    part2_ctrl_group = parser.add_argument_group('Part 2 Control (from YAML or CLI for Part 2)')
    part2_ctrl_group.add_argument('--visualize_pose', action=argparse.BooleanOptionalAction, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'visualize_pose', None), 
                        help='Override: Visualize ICP alignment.')
    part2_ctrl_group.add_argument('--visualize_pose_in_scene', action=argparse.BooleanOptionalAction, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'visualize_pose_in_scene', None),
                        help='Override: Visualize final transformed model in scene.')
    part2_ctrl_group.add_argument('--save_results', action=argparse.BooleanOptionalAction, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'save_results', None),
                        help='Override: Save estimated poses to a .npz file.')
    part2_ctrl_group.add_argument('--output_dir_part2', type=str, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'output_dir_part2', None), 
                        help='Override output directory for final poses. If None, uses a subfolder in intermediate_dir or Part1 output_dir.')

    # ICP Parameters (Now defined and used directly by Part 2, defaults from ICPParameters section of YAML)
    icp_params_group = parser.add_argument_group('ICP Parameters (from YAML or CLI for Part 2)')
    icp_params_group.add_argument('--icp_threshold', type=float, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_threshold', 20.0), # User changed this default
                                help='ICP max_correspondence_distance.')
    icp_params_group.add_argument('--icp_estimation_method', type=str, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_estimation_method', 'point_to_plane'), 
                                choices=['point_to_point', 'point_to_plane'], help="ICP estimation method.")
    icp_params_group.add_argument('--icp_relative_rmse', type=float, # Used as absolute RMSE change threshold
                                default=get_config_value(config_data, 'ICPParameters', 'icp_relative_rmse', 1e-6), # Adjusted default for more practical convergence
                                help='ICP convergence: absolute change in RMSE threshold.')
    # icp_relative_fitness is not directly used for convergence in this PyTorch ICP, but could be added.
    icp_params_group.add_argument('--icp_max_iter', type=int, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_max_iter', 50), # Reduced default for custom ICP
                                help='ICP convergence: max iterations.')
    icp_params_group.add_argument('--icp_min_points', type=int, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_min_points', 100),
                                help='Min instance points required for ICP processing.')

    cli_args_part2 = parser.parse_args(remaining_argv) 

    if cli_args_part2.intermediate_dir is None or cli_args_part2.intermediate_dir == '':
        parser.error("the following arguments are required: --intermediate_dir (must be provided via CLI or YAML config)")

    main(cli_args_part2) 