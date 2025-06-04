import cv2
import numpy as np
import open3d as o3d
import torch # For YOLO
from ultralytics import YOLO # YOLOv8
import subprocess
import os
import sys
import time
import tempfile
import re 
import yaml
import json

# Assuming orbbec_camera.py is in 'camera' subdirectory
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers
except ImportError as e:
    print(f"ERROR: Failed to import OrbbecCamera module: {e}")
    print("Please ensure 'camera' folder with 'orbbec_camera.py' is in the same directory or adjust PYTHONPATH.")
    sys.exit(1)

# --- USER CONFIGURATIONS ---
YOLO_MODEL_PATH = 'best.pt'  # Path to your YOLO model (e.g., yolov8n.pt, yolov8s.pt, or custom_model.pt)
# List of target class names that your YOLO model can detect AND you want to process
YOLO_TARGET_CLASS_NAMES = ['costa'] # IMPORTANT: Adjust to your model's classes and your targets
YOLO_CONF_THRESHOLD = 0.3  # Confidence threshold for YOLO detections

# IMPORTANT: Map detected class names to their corresponding CAD model files for Part 1
# The keys MUST match the names in YOLO_TARGET_CLASS_NAMES and your YOLO model output
CLASS_TO_MODEL_FILE_MAP = {
    'costa': 'Coca_Cola_250ml.STL',       # Replace with actual path
}

# PART1_SCRIPT_NAME = "_estimate_pose_part1_pytorch_coarse_align.py" # No longer calling Part 1
PART2_SCRIPT_NAME = "_estimate_pose_part2_icp_estimation.py"
CONFIG_YAML_FILE = "pose_estimation_config.yaml"

ORCHESTRATOR_BASE_OUTPUT_DIR = "./yolo_orchestrator_direct_to_part2_runs"
MODEL_SAMPLE_POINTS_FOR_PART2 = 2048 # Points to sample from CAD model for Part2's target
# ---

# --- New Helper Function for Median ROI Depth ---
def get_median_roi_depth_m(depth_frame_obj, roi_xyxy):
    """
    Calculates the median depth (in meters) of a given ROI from an Orbbec DepthFrame.

    Args:
        depth_frame_obj: The Orbbec SDK DepthFrame object.
        roi_xyxy: A tuple (xmin, ymin, xmax, ymax) defining the ROI.

    Returns:
        float: The median depth in meters, or None if ROI is invalid or no valid depth found.
    """
    if depth_frame_obj is None:
        print("Warning: get_median_roi_depth_m received None depth_frame_obj.")
        return None

    try:
        h = depth_frame_obj.get_height()
        w = depth_frame_obj.get_width()
        depth_scale = depth_frame_obj.get_depth_scale() # e.g., 0.001 for mm to m
        
        depth_data_u16 = np.frombuffer(depth_frame_obj.get_data(), dtype=np.uint16).reshape((h, w))

        xmin, ymin, xmax, ymax = roi_xyxy
        # Ensure ROI coordinates are integers and within frame bounds
        xmin = int(max(0, np.floor(xmin)))
        ymin = int(max(0, np.floor(ymin)))
        xmax = int(min(w, np.ceil(xmax)))
        ymax = int(min(h, np.ceil(ymax)))

        if xmin >= xmax or ymin >= ymax:
            # print(f"Warning: Invalid ROI for depth calculation: x ({xmin}-{xmax}), y ({ymin}-{ymax})")
            return None 

        roi_depth_u16_values = depth_data_u16[ymin:ymax, xmin:xmax]
        valid_depths_u16 = roi_depth_u16_values[roi_depth_u16_values > 0]  # Filter out zero (invalid) depths

        if valid_depths_u16.size == 0:
            # print(f"Warning: No valid depth points found in ROI: x ({xmin}-{xmax}), y ({ymin}-{ymax})")
            return None

        median_depth_u16 = np.median(valid_depths_u16)
        median_depth_m = median_depth_u16 * depth_scale
        # print(f"Median depth in ROI [{xmin}:{xmax},{ymin}:{ymax}]: {median_depth_m:.3f}m (from {median_depth_u16} scaled by {depth_scale})")
        return median_depth_m
    except Exception as e:
        print(f"Error in get_median_roi_depth_m: {e}")
        return None
# --- End New Helper Function ---

def find_script_path(script_name):
    # (Copied from previous orchestrator, ensures scripts/configs can be found)
    path_in_cwd = os.path.join(os.getcwd(), script_name)
    if os.path.exists(path_in_cwd) and os.path.isfile(path_in_cwd):
        return os.path.abspath(path_in_cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_in_script_dir = os.path.join(script_dir, script_name)
    if os.path.exists(path_in_script_dir) and os.path.isfile(path_in_script_dir):
        return os.path.abspath(path_in_script_dir)
    if script_name.endswith(".py"):
        print(f"Warning: Script {script_name} not found in standard locations. Assuming it's in PATH.")
        return script_name 
    print(f"ERROR: File {script_name} not found.")
    return None

def run_command_and_parse_output(command_list, parse_regex_dict, timeout_seconds=300):
    # (Copied from previous orchestrator)
    print(f"Executing command: {' '.join(command_list)}")
    results = {}
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        if process.returncode != 0:
            print(f"ERROR executing command: {' '.join(command_list)}")
            print(f"Return code: {process.returncode}")
            print("Stdout:"); print(stdout)
            print("Stderr:"); print(stderr)
            return None
        print("Command executed successfully. Stdout:"); print(stdout)
        if stderr: print("Stderr (warnings/minor errors):"); print(stderr)
        for key, pattern in parse_regex_dict.items():
            match = re.search(pattern, stdout)
            if match and match.group(1):
                results[key] = match.group(1).strip()
                print(f"  Parsed '{key}': {results[key]}")
            else:
                print(f"  Warning: Could not parse '{key}' from stdout using regex '{pattern}'.")
                results[key] = None
        return results
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out: {' '.join(command_list)}")
        if 'process' in locals() and process: process.kill(); process.communicate()
        return None
    except FileNotFoundError:
        print(f"ERROR: Command/script not found: {command_list[0]}. Check path and environment.")
        return None
    except Exception as e:
        print(f"ERROR running command {' '.join(command_list)}: {e}")
        return None

# --- Modified point cloud cropping function ---
def crop_point_cloud_from_roi_and_depth_range(
    full_pcd_o3d, roi_xyxy, color_cam_intrinsics_o3d, 
    min_depth_m, max_depth_m
):
    """
    Crops an Open3D point cloud based on a 2D ROI in the color image and a depth range.
    Assumes full_pcd_o3d points are in the color camera's 3D coordinate system.
    roi_xyxy: (xmin, ymin, xmax, ymax) for the bounding box.
    color_cam_intrinsics_o3d: Open3D PinholeCameraIntrinsic object for the color camera.
    min_depth_m, max_depth_m: Min and max depth values (in meters) for filtering.
    """
    if not full_pcd_o3d.has_points():
        print("Warning: Full point cloud is empty for depth range crop.")
        return o3d.geometry.PointCloud()

    points_np = np.asarray(full_pcd_o3d.points)
    colors_np = np.asarray(full_pcd_o3d.colors) if full_pcd_o3d.has_colors() else None

    # 1. Filter by depth (Z-coordinate in camera space)
    z_coords = points_np[:, 2]
    depth_filter_mask = (z_coords >= min_depth_m) & (z_coords <= max_depth_m)

    if not np.any(depth_filter_mask):
        print(f"Warning: No points found in depth range [{min_depth_m:.2f}m - {max_depth_m:.2f}m].")
        return o3d.geometry.PointCloud()

    points_depth_filtered = points_np[depth_filter_mask]
    colors_depth_filtered = colors_np[depth_filter_mask] if colors_np is not None and colors_np.shape[0] == points_np.shape[0] else None
    
    if points_depth_filtered.shape[0] == 0: # Should be caught by np.any above, but double check
        print("Warning: Zero points after depth filtering (unexpected).")
        return o3d.geometry.PointCloud()

    # 2. Filter by 2D ROI projection (from the depth-filtered points)
    fx = color_cam_intrinsics_o3d.intrinsic_matrix[0, 0]
    fy = color_cam_intrinsics_o3d.intrinsic_matrix[1, 1]
    cx = color_cam_intrinsics_o3d.intrinsic_matrix[0, 2]
    cy = color_cam_intrinsics_o3d.intrinsic_matrix[1, 2]

    X = points_depth_filtered[:, 0]
    Y = points_depth_filtered[:, 1]
    Z_proj = points_depth_filtered[:, 2] # Use Z from depth-filtered points for projection

    # Avoid division by zero for points at camera origin
    # Z_proj should be > 0 due to depth filtering (min_depth_m usually > 0)
    # but add a small epsilon for safety if min_depth_m can be very close to zero
    valid_projection_mask = Z_proj > 1e-6 
    if not np.any(valid_projection_mask):
        print("Warning: No points with Z > 0 for projection after depth filtering.")
        return o3d.geometry.PointCloud()

    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    # Perform projection only for points with valid Z
    u[valid_projection_mask] = (X[valid_projection_mask] * fx / Z_proj[valid_projection_mask]) + cx
    v[valid_projection_mask] = (Y[valid_projection_mask] * fy / Z_proj[valid_projection_mask]) + cy
    
    xmin_roi, ymin_roi, xmax_roi, ymax_roi = roi_xyxy
    
    # Create mask for points within ROI, considering only those valid for projection
    roi_mask_2d = (u >= xmin_roi) & (u < xmax_roi) & (v >= ymin_roi) & (v < ymax_roi) & valid_projection_mask
    
    final_points = points_depth_filtered[roi_mask_2d]

    if final_points.shape[0] == 0:
        print(f"Warning: No points found within the 2D ROI [{xmin_roi}-{xmax_roi}, {ymin_roi}-{ymax_roi}] after depth filtering.")
        return o3d.geometry.PointCloud()

    cropped_pcd_o3d = o3d.geometry.PointCloud()
    cropped_pcd_o3d.points = o3d.utility.Vector3dVector(final_points)

    if colors_depth_filtered is not None and colors_depth_filtered.shape[0] == points_depth_filtered.shape[0]:
        final_colors = colors_depth_filtered[roi_mask_2d]
        if final_colors.shape[0] == final_points.shape[0]:
             cropped_pcd_o3d.colors = o3d.utility.Vector3dVector(final_colors)
    
    print(f"Cropped point cloud to {len(cropped_pcd_o3d.points)} points from ROI and depth range [{min_depth_m:.2f}m - {max_depth_m:.2f}m].")
    return cropped_pcd_o3d
# --- End modified cropping function ---

def prepare_intermediate_data_for_part2(base_dir, object_pcd_o3d, cad_model_path, orchestrator_args_dict, full_scene_pcd_o3d):
    """
    Prepares the directory and files that Part 2 expects, without running Part 1.
    Saves: 
    - args.json (from orchestrator_args_dict)
    - scene_observed_for_icp.pcd (the cropped object_pcd_o3d)
    - instance_0_preprocessed.pcd (also the cropped object_pcd_o3d, for Part 2's glob pattern)
    - target_model_centered_for_icp.pcd -> RENAMED TO common_target_model_centered.pcd
    - common_target_model_original_scale.pcd
    - common_target_centroid_original_model_scale.npy
    - model_file_path.txt
    - initial_transform_for_icp.npy (identity matrix)
    - common_original_scene.pcd (the full scene point cloud)
    Returns the path to this prepared intermediate directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    print(f"Preparing intermediate data for Part 2 in: {base_dir}")

    # 1. Save args.json (can be minimal or based on orchestrator settings)
    args_file_path = os.path.join(base_dir, "args.json")
    try:
        with open(args_file_path, 'w') as f:
            json.dump(orchestrator_args_dict, f, indent=4)
        print(f"  Saved orchestrator args to {args_file_path}")
    except Exception as e:
        print(f"  ERROR saving args.json: {e}"); return None

    # 2. Save scene_observed_for_icp.pcd (the cropped object)
    path_scene_observed_for_icp = os.path.join(base_dir, "scene_observed_for_icp.pcd")
    try:
        o3d.io.write_point_cloud(path_scene_observed_for_icp, object_pcd_o3d)
        print(f"  Saved observed object (for ICP source) to: {path_scene_observed_for_icp}")
    except Exception as e:
        print(f"  ERROR saving scene_observed_for_icp.pcd: {e}"); return None
        
    # 2b. Save the same cropped object PCD as instance_0_preprocessed.pcd for Part 2
    path_instance_0_preprocessed = os.path.join(base_dir, "instance_0_preprocessed.pcd")
    try:
        o3d.io.write_point_cloud(path_instance_0_preprocessed, object_pcd_o3d)
        print(f"  Saved observed object also as instance_0_preprocessed.pcd to: {path_instance_0_preprocessed}")
    except Exception as e:
        print(f"  ERROR saving instance_0_preprocessed.pcd: {e}"); return None

    # 2c. Calculate and save centroid of the object_pcd_o3d (instance_0)
    path_instance_0_centroid = os.path.join(base_dir, "instance_0_centroid.npy")
    try:
        if object_pcd_o3d.has_points():
            instance_0_centroid_np = object_pcd_o3d.get_center()
            np.save(path_instance_0_centroid, instance_0_centroid_np)
            print(f"  Saved instance_0_centroid.npy to: {path_instance_0_centroid}")
        else:
            print("  Warning: object_pcd_o3d is empty, cannot save instance_0_centroid.npy. Part 2 might fail.")
            # Optionally save a zero centroid if Part 2 requires the file to exist
            # np.save(path_instance_0_centroid, np.array([0.0, 0.0, 0.0])) 
    except Exception as e:
        print(f"  ERROR saving instance_0_centroid.npy: {e}"); return None

    # 3. Load, process, and save target CAD model files
    if not os.path.exists(cad_model_path):
        print(f"  ERROR: CAD model file for Part 2 target not found: {cad_model_path}"); return None
    
    target_pcd_original_model_scale_o3d = o3d.geometry.PointCloud()
    try:
        temp_mesh = o3d.io.read_triangle_mesh(cad_model_path)
        if temp_mesh.has_vertices():
            target_pcd_original_model_scale_o3d = temp_mesh.sample_points_uniformly(MODEL_SAMPLE_POINTS_FOR_PART2)
        else: # Fallback if it's a PCD or other non-mesh format that sample_points_uniformly can't handle directly
            target_pcd_original_model_scale_o3d = o3d.io.read_point_cloud(cad_model_path)
            # If it was a PCD and we want to ensure a specific number of points, resample if necessary
            if len(target_pcd_original_model_scale_o3d.points) != MODEL_SAMPLE_POINTS_FOR_PART2 and len(target_pcd_original_model_scale_o3d.points) > 0 :
                 print(f"  Resampling loaded target PCD from {len(target_pcd_original_model_scale_o3d.points)} to {MODEL_SAMPLE_POINTS_FOR_PART2} points for consistency.")
                 target_pcd_original_model_scale_o3d = target_pcd_original_model_scale_o3d.random_down_sample(MODEL_SAMPLE_POINTS_FOR_PART2 / len(target_pcd_original_model_scale_o3d.points)) # This might not give exact number
                 # A more robust way for exact number if needed:
                 # target_pcd_original_model_scale_o3d = target_pcd_original_model_scale_o3d.uniform_down_sample(every_k_points) or a custom sampling.
                 # For now, let's assume read_point_cloud + possible uniform sampling (if it were mesh) is sufficient or direct use if it's already a pcd.
                 # The original code implies sample_points_uniformly is the main path for meshes.
                 # If it's already a PCD, we assume it's pre-sampled or the count doesn't strictly need to be MODEL_SAMPLE_POINTS_FOR_PART2 for the *original_scale* one.
                 # The *centered* one below IS sampled to MODEL_SAMPLE_POINTS_FOR_PART2.
    
        if not target_pcd_original_model_scale_o3d.has_points():
            raise ValueError("CAD model has no points after loading/sampling for Part 2 target.")
        
        target_centroid_original_np = target_pcd_original_model_scale_o3d.get_center()
        
        # Ensure the centered model FOR ICP has the specified number of points
        # If the original was a mesh, sample_points_uniformly already did this.
        # If it was a PCD, sample it now to ensure the correct number for target_pcd_centered_for_icp_o3d
        if len(target_pcd_original_model_scale_o3d.points) != MODEL_SAMPLE_POINTS_FOR_PART2 and temp_mesh.has_vertices(): # Mesh path was okay
             pass # Already sampled to MODEL_SAMPLE_POINTS_FOR_PART2 by sample_points_uniformly
        elif len(target_pcd_original_model_scale_o3d.points) == 0:
             raise ValueError("Target PCD is empty before centering and sampling.")
        else: # Was a PCD or sampling didn't give exact number, ensure target_pcd_centered_for_icp_o3d has correct count
            # Re-sample from the original scale PCD to get the target number of points for the centered one
            if len(target_pcd_original_model_scale_o3d.points) >= MODEL_SAMPLE_POINTS_FOR_PART2:
                print(f"  Sampling original target scale PCD from {len(target_pcd_original_model_scale_o3d.points)} to {MODEL_SAMPLE_POINTS_FOR_PART2} points for the centered ICP target.")
                # Farthest point sampling is good for uniform coverage
                target_pcd_for_centering_and_icp = target_pcd_original_model_scale_o3d.farthest_point_down_sample(MODEL_SAMPLE_POINTS_FOR_PART2)
            else: # Less points than required, use all of them
                print(f"  Warning: Original target scale PCD has less than {MODEL_SAMPLE_POINTS_FOR_PART2} points (has {len(target_pcd_original_model_scale_o3d.points)}). Using all available for centered ICP target.")
                target_pcd_for_centering_and_icp = o3d.geometry.PointCloud(target_pcd_original_model_scale_o3d) # Copy
        
        if not target_pcd_for_centering_and_icp.has_points():
             raise ValueError("Target PCD for centering and ICP is empty after sampling attempt.")

        target_pcd_centered_for_icp_o3d = o3d.geometry.PointCloud(target_pcd_for_centering_and_icp) # Make a copy to translate
        target_pcd_centered_for_icp_o3d.translate(-target_pcd_centered_for_icp_o3d.get_center()) # Center the sampled one

        path_common_target_orig_scale = os.path.join(base_dir, "common_target_model_original_scale.pcd")
        path_common_target_centroid = os.path.join(base_dir, "common_target_centroid_original_model_scale.npy")
        path_model_file_txt = os.path.join(base_dir, "model_file_path.txt")
        # path_target_centered_for_icp = os.path.join(base_dir, "target_model_centered_for_icp.pcd") # OLD NAME
        path_target_centered_for_icp = os.path.join(base_dir, "common_target_model_centered.pcd") # NEW NAME for Part 2

        o3d.io.write_point_cloud(path_common_target_orig_scale, target_pcd_original_model_scale_o3d)
        np.save(path_common_target_centroid, target_centroid_original_np) # This should be centroid of original scale
        with open(path_model_file_txt, 'w') as f_model: f_model.write(cad_model_path)
        o3d.io.write_point_cloud(path_target_centered_for_icp, target_pcd_centered_for_icp_o3d)
        print(f"  Saved target model files (original, centroid, centered for ICP as common_target_model_centered.pcd) to {base_dir}")

    except Exception as e_load_cad:
        print(f"  ERROR processing CAD model '{cad_model_path}' for Part 2: {e_load_cad}"); return None

    # 4. Save initial_transform_for_icp.npy (identity matrix as Part 1 is skipped)
    # RENAMING this to instance_0_pca_transform.npy for Part 2 consistency
    initial_transform_np = np.identity(4)
    # path_initial_transform = os.path.join(base_dir, "initial_transform_for_icp.npy") # OLD NAME
    path_instance_0_pca_transform = os.path.join(base_dir, "instance_0_pca_transform.npy") # NEW NAME
    try:
        np.save(path_instance_0_pca_transform, initial_transform_np)
        print(f"  Saved identity initial transform as instance_0_pca_transform.npy to: {path_instance_0_pca_transform}")
    except Exception as e_save_transform:
        print(f"  ERROR saving instance_0_pca_transform.npy: {e_save_transform}"); return None
        
    # 5. Save common_original_scene.pcd (full scene point cloud)
    path_common_original_scene = os.path.join(base_dir, "common_original_scene.pcd")
    try:
        if full_scene_pcd_o3d is not None and full_scene_pcd_o3d.has_points():
            o3d.io.write_point_cloud(path_common_original_scene, full_scene_pcd_o3d)
            print(f"  Saved full original scene to: {path_common_original_scene}")
        else:
            print("  Warning: Full scene PCD not provided or is empty. common_original_scene.pcd will not be saved.")
            # Create an empty file if Part 2 absolutely requires it to exist
            # o3d.io.write_point_cloud(path_common_original_scene, o3d.geometry.PointCloud()) 
    except Exception as e_save_scene:
        print(f"  ERROR saving common_original_scene.pcd: {e_save_scene}")
        # Do not return None here, as other files might be okay. Part 2 might handle missing scene.
    
    return base_dir # Return the path to the prepared directory

def main_yolo_orchestrator():
    print("YOLO Orchestrator Starting (Direct to Part 2 Mode)...")
    print(f"Attempting to load YOLO model from: {YOLO_MODEL_PATH}")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model from '{YOLO_MODEL_PATH}'. Error: {e}")
        print("Please ensure the model path is correct and Ultralytics YOLO is installed ('pip install ultralytics').")
        return

    print("Initializing Orbbec Camera...")
    camera_instance = None
    try:
        available_sns = get_serial_numbers()
        if not available_sns:
            print("ERROR: No Orbbec devices found. Please check connection."); return
        print(f"Found Orbbec devices: {available_sns}. Using first one: {available_sns[0]}")
        camera_instance = OrbbecCamera(available_sns[0])
        camera_instance.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        if camera_instance.param is None or camera_instance.param.rgb_intrinsic is None:
            raise RuntimeError("Failed to get RGB camera intrinsics from camera after starting stream.")
        print("Orbbec Camera initialized and stream started with D2C alignment.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Orbbec Camera or start stream: {e}")
        if camera_instance: camera_instance.stop()
        return

    os.makedirs(ORCHESTRATOR_BASE_OUTPUT_DIR, exist_ok=True)
    # part1_script_path = find_script_path(PART1_SCRIPT_NAME) # No longer needed
    part2_script_path = find_script_path(PART2_SCRIPT_NAME)
    config_yaml_path = find_script_path(CONFIG_YAML_FILE)

    if not all([part2_script_path, config_yaml_path]): # Check only Part2 script and config
        print("ERROR: Part 2 script or main config file not found. Exiting.")
        if camera_instance: camera_instance.stop(); return

    try:
        color_width = camera_instance.color_profile.get_width()
        color_height = camera_instance.color_profile.get_height()
        o3d_rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=color_width, height=color_height,
            fx=camera_instance.param.rgb_intrinsic.fx, fy=camera_instance.param.rgb_intrinsic.fy,
            cx=camera_instance.param.rgb_intrinsic.cx, cy=camera_instance.param.rgb_intrinsic.cy
        )
        print("Successfully created Open3D RGB intrinsics object.")
    except Exception as e:
        print(f"ERROR: Could not get camera profile info or create O3D intrinsics: {e}")
        if camera_instance: camera_instance.stop(); return

    # --- Load YAML Configuration for ROI Depth Offsets ---
    config_data = {}
    if os.path.exists(CONFIG_YAML_FILE):
        try:
            with open(CONFIG_YAML_FILE, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Successfully loaded orchestrator config from {CONFIG_YAML_FILE}")
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML config file {CONFIG_YAML_FILE}: {e}. Using default ROI offsets.")
        except Exception as e:
            print(f"Warning: Could not read YAML config file {CONFIG_YAML_FILE}: {e}. Using default ROI offsets.")
            
    roi_extraction_config = config_data.get('ROIExtraction', {})
    # Define these variables in the function scope before the loop
    global DEPTH_BEHIND_OFFSET_M, DEPTH_FRONT_OFFSET_M # Make them global if used in other functions not receiving them as params, though it seems they are only used locally in main_yolo_orchestrator
    DEPTH_BEHIND_OFFSET_M = roi_extraction_config.get('depth_behind_offset_m', 0.5) # Default 0.5m
    DEPTH_FRONT_OFFSET_M = roi_extraction_config.get('depth_front_offset_m', 0.05)  # Default 0.05m (small positive value)
    print(f"ROI Depth Offsets: Behind={DEPTH_BEHIND_OFFSET_M}m, Front={DEPTH_FRONT_OFFSET_M}m")
    # --- End YAML Loading ---

    print("\nOrchestrator ready. Press 'r' in OpenCV window to detect objects and run pose estimation pipeline.")
    print("Press 'q' to quit.")

    # Create a dictionary of orchestrator settings that might be useful for args.json
    # This is a placeholder; Part 2 will load its own args from YAML and CLI overrides
    orchestrator_args_for_part2_json = {
        "yolo_model": YOLO_MODEL_PATH,
        "yolo_target_classes": YOLO_TARGET_CLASS_NAMES,
        "invoked_by_yolo_orchestrator_direct_to_part2": True
        # Add any other relevant info if Part 2 needs to know about its caller
    }

    try:
        while True:
            color_image, _, depth_frame_obj = camera_instance.get_frames()
            if color_image is None:
                cv2.imshow("YOLO Object Detection (Orbbec)", np.zeros((480,640,3), dtype=np.uint8))
                if cv2.waitKey(30) & 0xFF == ord('q'): break
                continue
            
            display_image = color_image.copy()
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'): print("'q' pressed. Exiting..."); break
            
            if key == ord('r'):
                print("\n'r' pressed. Running YOLO detection and pose estimation pipeline (direct to Part 2)...")
                yolo_results = yolo_model(color_image, conf=YOLO_CONF_THRESHOLD,show=True)
                processed_one_object = False
                if yolo_results and len(yolo_results) > 0:
                    detections = yolo_results[0]
                    names = detections.names
                    print(f"YOLO found {len(detections.boxes)} objects in total.")
                    target_object_found_this_trigger = False

                    for i in range(len(detections.boxes)):
                        box = detections.boxes[i]
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = names.get(cls_id, f"ClassID_{cls_id}")
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                        if conf >= YOLO_CONF_THRESHOLD:
                            is_target_class = class_name in YOLO_TARGET_CLASS_NAMES
                            color = (0, 255, 0) if is_target_class else (255, 0, 0) # Green for target, Blue for non-target

                            cv2.rectangle(display_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                            cv2.putText(display_image, f"{class_name} {conf:.2f}", (xyxy[0], xyxy[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            if is_target_class:
                                print(f"  Target object for processing: {class_name} with confidence {conf:.2f}")
                                target_object_found_this_trigger = True
                                
                                # --- Get representative depth of the object in the ROI ---
                                object_median_depth_m = None
                                if depth_frame_obj: # Ensure depth_frame_obj is valid
                                    object_median_depth_m = get_median_roi_depth_m(depth_frame_obj, xyxy)
                                
                                if object_median_depth_m is None:
                                    print(f"  Warning: Could not determine median depth for ROI of {class_name}. Skipping depth-based crop modification for this object.")
                                    # Fallback to simple 2D ROI crop (or skip entirely if depth is crucial)
                                    # For now, let's try to get full PCD and then if depth crop fails, it will be handled.
                                    # If we skip here, the else block for no_target_found won't be triggered correctly.

                                # --- Get full point cloud ---
                                print("  Getting full point cloud...")
                                raw_data_from_camera = camera_instance.get_point_cloud(colored=True) # This applies its own MIN/MAX_DEPTH filter
                                
                                # --- Ensure full_pcd_o3d is an Open3D PointCloud object ---
                                if raw_data_from_camera is None:
                                    print("  Warning: Camera returned None for full point cloud. Using empty point cloud.")
                                    full_pcd_o3d = o3d.geometry.PointCloud()
                                elif isinstance(raw_data_from_camera, np.ndarray):
                                    if raw_data_from_camera.ndim == 2 and raw_data_from_camera.shape[0] > 0 and raw_data_from_camera.shape[1] >= 3:
                                        full_pcd_o3d = o3d.geometry.PointCloud()
                                        full_pcd_o3d.points = o3d.utility.Vector3dVector(raw_data_from_camera[:, :3])
                                        if raw_data_from_camera.shape[1] >= 6: # Check for color
                                            colors = raw_data_from_camera[:, 3:6]
                                            if colors.dtype == np.uint8:
                                                colors = colors.astype(np.float64) / 255.0
                                            elif colors.dtype in [np.float32, np.float64]:
                                                if np.any(colors < 0.0) or np.any(colors > 1.0):
                                                    # print(f"  Warning: Full PCD float color data appears outside [0,1]. Clipping.")
                                                    colors = np.clip(colors, 0.0, 1.0)
                                            full_pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
                                    else:
                                        print(f"  Warning: Full PCD NumPy array from camera has invalid shape: {raw_data_from_camera.shape}. Using empty point cloud.")
                                        full_pcd_o3d = o3d.geometry.PointCloud()
                                elif hasattr(raw_data_from_camera, 'has_points'): # Already an Open3D PointCloud
                                    full_pcd_o3d = raw_data_from_camera
                                else:
                                    print(f"  Warning: Camera returned unexpected type for full point cloud: {type(raw_data_from_camera)}. Using empty point cloud.")
                                    full_pcd_o3d = o3d.geometry.PointCloud()
                                # --- End Open3D conversion ---
                                
                                if full_pcd_o3d is None or not full_pcd_o3d.has_points():
                                     # This check was already here, but now apply it after attempting to get full_pcd_o3d
                                    print("  ERROR: Failed to get a valid full point cloud. Skipping this object.")
                                    continue # Skip to next detected box or next frame trigger

                                print(f"  Full point cloud has {len(full_pcd_o3d.points)} points (after camera's initial depth filter).")

                                # --- Crop point cloud using ROI and Depth Range ---
                                if object_median_depth_m is not None:
                                    min_crop_depth_m = object_median_depth_m - DEPTH_FRONT_OFFSET_M
                                    max_crop_depth_m = object_median_depth_m + DEPTH_BEHIND_OFFSET_M
                                    print(f"  Target object median depth: {object_median_depth_m:.3f}m. Cropping PC in range: [{min_crop_depth_m:.3f}m - {max_crop_depth_m:.3f}m].")
                                    
                                    object_pcd_o3d = crop_point_cloud_from_roi_and_depth_range(
                                        full_pcd_o3d, xyxy, o3d_rgb_intrinsics,
                                        min_crop_depth_m, max_crop_depth_m
                                    )
                                else:
                                    # Fallback: if median depth couldn't be found, try a wide depth range or old 2D method
                                    print("  Fallback: Using a wide depth range for cropping as object median depth was not found.")
                                    # This will use a very generic depth range, effectively just the 2D ROI if min/max are too broad.
                                    # Or, revert to the old 2D-only cropping method if preferred.
                                    # For now, let's use very broad min/max which the existing crop_point_cloud_from_roi_and_depth_range
                                    # will mostly ignore if points are outside the camera's own MIN_DEPTH/MAX_DEPTH from get_point_cloud()
                                    object_pcd_o3d = crop_point_cloud_from_roi_and_depth_range(
                                        full_pcd_o3d, xyxy, o3d_rgb_intrinsics,
                                        0.1, 10.0 # Default very wide range (0.1m to 10m)
                                    )


                                if not object_pcd_o3d.has_points() or len(object_pcd_o3d.points) < 10: # Check if cropping yielded enough points
                                    print("  Warning: Cropped object point cloud (with depth range) is empty or too sparse. Skipping."); 
                                    continue # Skip to next detected box or next frame trigger
                                
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                current_run_output_dir = os.path.join(ORCHESTRATOR_BASE_OUTPUT_DIR, f"run_{class_name}_{timestamp}")
                                intermediate_dir_for_part2 = os.path.join(current_run_output_dir, "intermediate_for_part2") 
                                
                                target_cad_model_file = CLASS_TO_MODEL_FILE_MAP.get(class_name)
                                if not target_cad_model_file or not os.path.exists(target_cad_model_file):
                                    print(f"  ERROR: CAD Model for class '{class_name}' not found. Path: '{target_cad_model_file}'. Skipping."); continue
                                
                                # Prepare the intermediate directory for Part 2 manually
                                prepared_intermediate_path = prepare_intermediate_data_for_part2(
                                    intermediate_dir_for_part2,
                                    object_pcd_o3d,              # This is the cropped ROI point cloud
                                    target_cad_model_file,
                                    orchestrator_args_for_part2_json,
                                    full_pcd_o3d                 # Pass the full scene PCD
                                )
                                if not prepared_intermediate_path:
                                    print("  ERROR: Failed to prepare intermediate data for Part 2. Skipping."); continue
                                
                                print(f"\n  --- Running Part 2 for {class_name} (using prepared data: {prepared_intermediate_path}) ---")
                                part2_final_results_dir = os.path.join(current_run_output_dir, "part2_final_poses")
                                os.makedirs(part2_final_results_dir, exist_ok=True)
                                cmd_part2 = [
                                    sys.executable, part2_script_path,
                                    "--intermediate_dir", prepared_intermediate_path,
                                    "--config_file", config_yaml_path,
                                    "--output_dir_part2", part2_final_results_dir,
                                ]
                                cmd_part2.append("--visualize_pose")
                                cmd_part2.append("--visualize_pose_in_scene")
                                cmd_part2.append("--save_results")

                                parse_dict_part2 = {"npz_file": r"Saved PyTorch ICP poses to (.*\.npz)"}
                                part2_results = run_command_and_parse_output(cmd_part2, parse_dict_part2)

                                if not part2_results or not part2_results.get("npz_file"):
                                    print("  ERROR: Part 2 failed or could not parse NPZ file path."); continue
                                npz_file_path = part2_results["npz_file"]
                                if not os.path.exists(npz_file_path):
                                    print(f"  ERROR: NPZ file reported by Part 2 does not exist: {npz_file_path}"); continue
                                
                                print(f"\n  --- Results for {class_name} from {npz_file_path} ---")
                                try:
                                    estimated_poses_data = np.load(npz_file_path)
                                    if not estimated_poses_data.files:
                                        print("  Warning: NPZ file is empty (no poses saved by Part 2).")
                                    for instance_key in estimated_poses_data.files:
                                        pose_matrix = estimated_poses_data[instance_key]
                                        print(f"    Estimated Pose Matrix for '{instance_key}':\n{pose_matrix}")
                                except Exception as e_load_npz:
                                    print(f"  ERROR loading or processing NPZ file {npz_file_path}: {e_load_npz}")
                                
                                print(f"  Pipeline finished for detected object: {class_name}")
                                processed_one_object = True
                                break 
                    
                    if not target_object_found_this_trigger:
                        print("No target objects (from YOLO_TARGET_CLASS_NAMES) found with sufficient confidence.")
                else:
                    print("YOLO did not return any results for this frame.")
            
            cv2.imshow("YOLO Object Detection (Orbbec)", display_image)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting orchestrator...")
    finally:
        if camera_instance:
            print("Stopping Orbbec camera...")
            camera_instance.stop()
        cv2.destroyAllWindows()
        print("YOLO Orchestrator finished (Direct to Part 2 Mode).")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_YAML_FILE):
        print(f"ERROR: Main config file '{CONFIG_YAML_FILE}' not found."); sys.exit(1)
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERROR: YOLO model file '{YOLO_MODEL_PATH}' not found."); sys.exit(1)
    missing_models = False
    for cn in YOLO_TARGET_CLASS_NAMES:
        if cn not in CLASS_TO_MODEL_FILE_MAP or not CLASS_TO_MODEL_FILE_MAP[cn] or not os.path.exists(CLASS_TO_MODEL_FILE_MAP[cn]):
            print(f"ERROR: CAD model file for class '{cn}' is not defined or does not exist.")
            print(f"  Expected path: {CLASS_TO_MODEL_FILE_MAP.get(cn, 'Not specified')}")
            missing_models = True
    if missing_models: print("Please update CLASS_TO_MODEL_FILE_MAP."); sys.exit(1)

    main_yolo_orchestrator() 