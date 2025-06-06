import cv2
import numpy as np
import open3d as o3d
import torch # For YOLO
from ultralytics import YOLO # YOLOv8
import os
import sys
import time
import yaml
import json # Keep for args.json if any part of copied code implicitly uses it, though primary use is removed.
from pyorbbecsdk import OBAlignMode

# Assuming orbbec_camera.py is in 'camera' subdirectory
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers
except ImportError as e:
    print(f"ERROR: Failed to import OrbbecCamera module: {e}")
    print("Please ensure 'camera' folder with 'orbbec_camera.py' is in the same directory or adjust PYTHONPATH.")
    sys.exit(1)

# --- USER CONFIGURATIONS ---
YOLO_MODEL_PATH = 'yolov8m.pt'
YOLO_TARGET_CLASS_NAMES = ['truck','boat','remote'] # IMPORTANT: Adjust to your model's classes and your targets
YOLO_CONF_THRESHOLD = 0.3

# 模板原点处理配置
# 可选模式: "centroid", "bbox_center", "bottom_center", "bbox_bottom", "geometric_center", "none", "interactive"
TEMPLATE_ORIGIN_MODE = "interactive"  # 设置为 "interactive" 可以每次手动选择，或设置具体模式自动应用

# CLASS_TO_MODEL_FILE_MAP is not strictly needed for template creation,
# but kept to minimize changes from step2_yolo_orchestrator.py base.
# The class_name directly from YOLO will be used for the template filename.
CLASS_TO_MODEL_FILE_MAP = {
    # 'keyboard': 'lixun.STL', # Path not used, but key 'keyboard' is if in YOLO_TARGET_CLASS_NAMES
    'truck': 'lixun.STL',  # Path not used, but key 'keyboard' is if in YOLO_TARGET_CLASS_NAMES
    'boat': 'lixun.STL',  # Path not used, but key 'keyboard' is if in YOLO_TARGET_CLASS_NAMES
    'remote': 'lixun.STL',  # Path not used, but key 'keyboard' is if in YOLO_TARGET_CLASS_NAMES

}

CONFIG_YAML_FILE = "pose_estimation_config.yaml" # For ROIExtraction parameters
PCD_CLASS_TEMPLATES_DIR = "./pcd_class_templates"
# ---

# --- Helper Function for Median ROI Depth (Copied from step2_yolo_orchestrator.py) ---
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
        xmin = int(max(0, np.floor(xmin)))
        ymin = int(max(0, np.floor(ymin)))
        xmax = int(min(w, np.ceil(xmax)))
        ymax = int(min(h, np.ceil(ymax)))

        if xmin >= xmax or ymin >= ymax:
            return None 

        roi_depth_u16_values = depth_data_u16[ymin:ymax, xmin:xmax]
        valid_depths_u16 = roi_depth_u16_values[roi_depth_u16_values > 0]

        if valid_depths_u16.size == 0:
            return None

        median_depth_u16 = np.median(valid_depths_u16)
        median_depth_m = median_depth_u16 * depth_scale
        return median_depth_m
    except Exception as e:
        print(f"Error in get_median_roi_depth_m: {e}")
        return None
# --- End Median ROI Depth Helper ---

# --- Point cloud cropping function (Copied from step2_yolo_orchestrator.py) ---
def crop_point_cloud_from_roi_and_depth_range(
    full_pcd_o3d, roi_xyxy, color_cam_intrinsics_o3d, 
    min_depth_m, max_depth_m
):
    if not full_pcd_o3d.has_points():
        print("Warning: Full point cloud is empty for depth range crop.")
        return o3d.geometry.PointCloud()

    points_np = np.asarray(full_pcd_o3d.points)
    colors_np = np.asarray(full_pcd_o3d.colors) if full_pcd_o3d.has_colors() else None

    z_coords = points_np[:, 2]
    depth_filter_mask = (z_coords >= min_depth_m) & (z_coords <= max_depth_m)

    if not np.any(depth_filter_mask):
        print(f"Warning: No points found in depth range [{min_depth_m:.2f}m - {max_depth_m:.2f}m].")
        return o3d.geometry.PointCloud()

    points_depth_filtered = points_np[depth_filter_mask]
    colors_depth_filtered = colors_np[depth_filter_mask] if colors_np is not None and colors_np.shape[0] == points_np.shape[0] else None
    
    if points_depth_filtered.shape[0] == 0:
        print("Warning: Zero points after depth filtering (unexpected).")
        return o3d.geometry.PointCloud()

    fx = color_cam_intrinsics_o3d.intrinsic_matrix[0, 0]
    fy = color_cam_intrinsics_o3d.intrinsic_matrix[1, 1]
    cx = color_cam_intrinsics_o3d.intrinsic_matrix[0, 2]
    cy = color_cam_intrinsics_o3d.intrinsic_matrix[1, 2]

    X = points_depth_filtered[:, 0]
    Y = points_depth_filtered[:, 1]
    Z_proj = points_depth_filtered[:, 2]
    valid_projection_mask = Z_proj > 1e-6 
    if not np.any(valid_projection_mask):
        print("Warning: No points with Z > 0 for projection after depth filtering.")
        return o3d.geometry.PointCloud()

    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    u[valid_projection_mask] = (X[valid_projection_mask] * fx / Z_proj[valid_projection_mask]) + cx
    v[valid_projection_mask] = (Y[valid_projection_mask] * fy / Z_proj[valid_projection_mask]) + cy
    
    xmin_roi, ymin_roi, xmax_roi, ymax_roi = roi_xyxy
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
# --- End cropping function ---

def process_template_origin(pcd_o3d, origin_mode="centroid"):
    """
    修改点云模板的原点位置。
    
    Args:
        pcd_o3d: Open3D点云对象
        origin_mode: 原点调整模式，可选:
            - "centroid": 将质心移动到原点(0,0,0)
            - "bbox_center": 将边界盒中心移动到原点
            - "bottom_center": 将底面中心移动到原点，保持Z=0在底部
            - "bbox_bottom": 将边界盒底面中心移动到原点
            - "geometric_center": 将几何中心移动到原点
            - "none": 不做任何修改
    
    Returns:
        处理后的点云对象
    """
    if not pcd_o3d.has_points():
        print("警告: 点云为空，无法处理原点")
        return pcd_o3d
        
    points = np.asarray(pcd_o3d.points)
    
    if origin_mode == "centroid":
        # 计算质心并移动到原点
        centroid = np.mean(points, axis=0)
        print(f"  质心位置: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        points_adjusted = points - centroid
        print(f"  已将质心移动到原点")
        
    elif origin_mode == "bbox_center":
        # 计算边界盒中心并移动到原点
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        print(f"  边界盒中心位置: [{bbox_center[0]:.3f}, {bbox_center[1]:.3f}, {bbox_center[2]:.3f}]")
        points_adjusted = points - bbox_center
        print(f"  已将边界盒中心移动到原点")
        
    elif origin_mode == "bottom_center":
        # 将底面中心移动到原点，Z轴最小值移动到0
        z_min = np.min(points[:, 2])
        xy_center = np.mean(points[:, :2], axis=0)
        offset = np.array([xy_center[0], xy_center[1], z_min])
        print(f"  底面中心位置: [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")
        points_adjusted = points - offset
        print(f"  已将底面中心移动到原点，底面位于Z=0")
        
    elif origin_mode == "bbox_bottom":
        # 将边界盒底面中心移动到原点
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_bottom_center = np.array([(bbox_min[0] + bbox_max[0])/2, 
                                      (bbox_min[1] + bbox_max[1])/2, 
                                      bbox_min[2]])
        print(f"  边界盒底面中心: [{bbox_bottom_center[0]:.3f}, {bbox_bottom_center[1]:.3f}, {bbox_bottom_center[2]:.3f}]")
        points_adjusted = points - bbox_bottom_center
        print(f"  已将边界盒底面中心移动到原点")
        
    elif origin_mode == "geometric_center":
        # 几何中心 (与质心类似，但考虑几何分布)
        geometric_center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
        print(f"  几何中心位置: [{geometric_center[0]:.3f}, {geometric_center[1]:.3f}, {geometric_center[2]:.3f}]")
        points_adjusted = points - geometric_center
        print(f"  已将几何中心移动到原点")
        
    elif origin_mode == "none":
        print("  未修改原点位置")
        points_adjusted = points
        
    else:
        print(f"  警告: 未知的原点模式 '{origin_mode}'，使用默认质心模式")
        centroid = np.mean(points, axis=0)
        points_adjusted = points - centroid
        
    # 更新点云
    pcd_processed = o3d.geometry.PointCloud()
    pcd_processed.points = o3d.utility.Vector3dVector(points_adjusted)
    
    # 保持颜色信息
    if pcd_o3d.has_colors():
        pcd_processed.colors = pcd_o3d.colors
        
    return pcd_processed

def interactive_origin_selection():
    """
    交互式选择原点处理模式
    """
    print("\n=== 选择原点处理模式 ===")
    print("1. 质心对齐 (centroid) - 将点云质心移动到原点")
    print("2. 边界盒中心 (bbox_center) - 将边界盒中心移动到原点")  
    print("3. 底面中心 (bottom_center) - 将底面中心移动到原点，底面位于Z=0")
    print("4. 边界盒底面 (bbox_bottom) - 将边界盒底面中心移动到原点")
    print("5. 几何中心 (geometric_center) - 将几何中心移动到原点")
    print("6. 不修改 (none) - 保持原始位置")
    
    mode_map = {
        '1': 'centroid',
        '2': 'bbox_center', 
        '3': 'bottom_center',
        '4': 'bbox_bottom',
        '5': 'geometric_center',
        '6': 'none'
    }
    
    while True:
        try:
            choice = input("请选择 (1-6): ").strip()
            if choice in mode_map:
                return mode_map[choice]
            else:
                print("无效选择，请输入 1-6")
        except KeyboardInterrupt:
            print("\n使用默认质心对齐模式")
            return 'centroid'

def find_config_file_path(file_name):
    path_in_cwd = os.path.join(os.getcwd(), file_name)
    if os.path.exists(path_in_cwd) and os.path.isfile(path_in_cwd):
        return os.path.abspath(path_in_cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_in_script_dir = os.path.join(script_dir, file_name)
    if os.path.exists(path_in_script_dir) and os.path.isfile(path_in_script_dir):
        return os.path.abspath(path_in_script_dir)
    print(f"ERROR: Config file {file_name} not found in current working directory or script directory.")
    return None

def main_create_template():
    print("Create PCD Template from YOLO Starting...")
    print(f"Attempting to load YOLO model from: {YOLO_MODEL_PATH}")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model from '{YOLO_MODEL_PATH}'. Error: {e}")
        return

    print("Initializing Orbbec Camera...")
    camera_instance = None
    try:
        available_sns = get_serial_numbers()
        if not available_sns:
            print("ERROR: No Orbbec devices found. Please check connection."); return
        print(f"Found Orbbec devices: {available_sns}. Using first one: {available_sns[0]}")
        camera_instance = OrbbecCamera("CP1Z842000DL")
        camera_instance.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        if camera_instance.param is None or camera_instance.param.rgb_intrinsic is None:
            raise RuntimeError("Failed to get RGB camera intrinsics from camera after starting stream.")
        print("Orbbec Camera initialized and stream started with D2C alignment.")
        
        color_stream_is_active = camera_instance.color_profile is not None
        alignment_is_active = False
        if hasattr(camera_instance, 'config') and camera_instance.config is not None and hasattr(camera_instance.config, 'get_align_mode'):
            alignment_is_active = camera_instance.config.get_align_mode() != OBAlignMode.ALIGN_DISABLE
        print(f"DEBUG: Camera init done. Color stream: {color_stream_is_active}, Depth-to-Color alignment: {alignment_is_active}")

        if not camera_instance.stream:
            print("ERROR: Orbbec camera stream is not active after attempting to start. Exiting.")
            return

        os.makedirs(PCD_CLASS_TEMPLATES_DIR, exist_ok=True)
        
        config_yaml_abs_path = find_config_file_path(CONFIG_YAML_FILE)
        if not config_yaml_abs_path: return


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

        config_data = {}
        if os.path.exists(config_yaml_abs_path):
            try:
                with open(config_yaml_abs_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                print(f"Successfully loaded config from {config_yaml_abs_path}")
            except Exception as e:
                print(f"Warning: Error loading/parsing YAML {config_yaml_abs_path}: {e}. Using default ROI offsets.")
            
        roi_extraction_config = config_data.get('ROIExtraction', {})
        DEPTH_BEHIND_OFFSET_M = roi_extraction_config.get('depth_behind_offset_m', 0.5)
        DEPTH_FRONT_OFFSET_M = roi_extraction_config.get('depth_front_offset_m', 0.05)
        print(f"ROI Depth Offsets: Behind={DEPTH_BEHIND_OFFSET_M}m, Front={DEPTH_FRONT_OFFSET_M}m")

        print("模板创建器就绪。")
        print("操作说明:")
        print("  按 'd' - 检测对象并准备模板保存")
        print("  按 's' - 保存准备好的点云作为模板 (包含原点处理)")
        print("  按 'q' - 退出程序")
        print(f"当前原点处理模式: {TEMPLATE_ORIGIN_MODE}")
        if TEMPLATE_ORIGIN_MODE == "interactive":
            print("  (保存时将提示选择具体的原点处理方式)")

        current_object_pcd_for_template = None
        current_class_name_for_template = None

        try:
            while True:
                color_image, _, depth_frame_obj = camera_instance.get_frames()
                if color_image is None:
                    cv2.imshow("Create PCD Template (Orbbec)", np.zeros((480,640,3), dtype=np.uint8))
                    if cv2.waitKey(30) & 0xFF == ord('q'): break
                    continue
                
                display_image = color_image.copy()
                key = cv2.waitKey(10) & 0xFF

                if key == ord('q'): print("'q' pressed. Exiting..."); break
                
                if key == ord('d'): # Detect and prepare object for saving
                    print("'d' pressed. Running YOLO detection...")
                    yolo_results = yolo_model(color_image, conf=YOLO_CONF_THRESHOLD, verbose=False,show=True) # verbose=False for cleaner logs
                    current_object_pcd_for_template = None # Reset on new detection trigger
                    current_class_name_for_template = None

                    if yolo_results and len(yolo_results) > 0:
                        detections = yolo_results[0]
                        names = detections.names
                        print(f"YOLO found {len(detections.boxes)} objects in total.")
                        
                        for i in range(len(detections.boxes)): # Iterate to find first target
                            box = detections.boxes[i]
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = names.get(cls_id, f"ClassID_{cls_id}")
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)

                            if conf >= YOLO_CONF_THRESHOLD and class_name in YOLO_TARGET_CLASS_NAMES:
                                print(f"  Target object found: {class_name} (Conf: {conf:.2f}). Preparing point cloud...")
                                cv2.rectangle(display_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                                cv2.putText(display_image, f"CAPTURING: {class_name}", (xyxy[0], xyxy[1]-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                                object_median_depth_m = get_median_roi_depth_m(depth_frame_obj, xyxy)
                                if object_median_depth_m is None:
                                    print(f"  Warning: Could not get median depth for {class_name}. Full scene depth range will be used for crop.")
                                
                                print("  Getting full point cloud...")
                                raw_data_from_camera = camera_instance.get_point_cloud(colored=True)
                                
                                # --- Color Processing Logic (copied from step2_yolo_orchestrator.py) ---
                                full_pcd_o3d = o3d.geometry.PointCloud()
                                if raw_data_from_camera is not None and isinstance(raw_data_from_camera, np.ndarray) and raw_data_from_camera.ndim == 2 and raw_data_from_camera.shape[0] > 0 and raw_data_from_camera.shape[1] >= 3:
                                    full_pcd_o3d.points = o3d.utility.Vector3dVector(raw_data_from_camera[:, :3])
                                    if raw_data_from_camera.shape[1] >= 6:
                                        colors_np = raw_data_from_camera[:, 3:6]
                                        if colors_np.dtype == np.uint8:
                                            colors_np_float = colors_np.astype(np.float64) / 255.0
                                            full_pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np_float)
                                        elif colors_np.dtype in [np.float32, np.float64]:
                                            if np.any(colors_np < 0.0) or np.any(colors_np > 1.0):
                                                colors_np_normalized = np.clip(colors_np / 255.0, 0.0, 1.0)
                                                full_pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np_normalized)
                                            else:
                                                full_pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np)
                                # --- End Color Processing ---

                                if not full_pcd_o3d.has_points():
                                    print("  ERROR: Failed to get a valid full point cloud. Skipping template preparation for this object.")
                                    continue

                                min_crop_depth_m = 0.1 # Default wide range if median depth fails
                                max_crop_depth_m = 10.0
                                if object_median_depth_m is not None:
                                    min_crop_depth_m = object_median_depth_m - DEPTH_FRONT_OFFSET_M
                                    max_crop_depth_m = object_median_depth_m + DEPTH_BEHIND_OFFSET_M
                                    print(f"  Target object median depth: {object_median_depth_m:.3f}m. Cropping PC in range: [{min_crop_depth_m:.3f}m - {max_crop_depth_m:.3f}m].")
                                else:
                                     print(f"  Using default wide depth range for cropping: [{min_crop_depth_m:.2f}m - {max_crop_depth_m:.2f}m].")


                                temp_object_pcd = crop_point_cloud_from_roi_and_depth_range(
                                    full_pcd_o3d, xyxy, o3d_rgb_intrinsics,
                                    min_crop_depth_m, max_crop_depth_m
                                )

                                if temp_object_pcd.has_points() and len(temp_object_pcd.points) >= 10: # Min points check
                                    current_object_pcd_for_template = temp_object_pcd
                                    current_class_name_for_template = class_name
                                    print(f"  SUCCESS: Point cloud for '{class_name}' prepared ({len(temp_object_pcd.points)} points). Press 's' to save.")
                                    if temp_object_pcd.has_colors():
                                        print(f"  Prepared PCD has colors. First 3: {np.asarray(temp_object_pcd.colors)[:3]}")
                                    else:
                                        print("  Prepared PCD does NOT have colors.")
                                    # Visualize the prepared PCD
                                    o3d.visualization.draw_geometries([current_object_pcd_for_template], window_name=f"Preview: {class_name} Template")

                                else:
                                    print(f"  Warning: Cropped point cloud for '{class_name}' is empty or too sparse after filtering. Not prepared for saving.")
                                break # Process first detected target object and then wait for 's' or new 'd'
                            else: # Draw box for non-target or low confidence
                                 if conf >= YOLO_CONF_THRESHOLD : # Non-target but above threshold
                                    cv2.rectangle(display_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 1)
                                    cv2.putText(display_image, f"{class_name} {conf:.2f}", (xyxy[0], xyxy[1]-5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)


                        if not current_class_name_for_template: # If loop finishes and no target was prepared
                             print("No target objects from YOLO_TARGET_CLASS_NAMES captured in this detection cycle.")
                    else:
                        print("YOLO did not return any results for this frame on 'd' press.")

                elif key == ord('s'): # Save the prepared template
                    if current_object_pcd_for_template and current_class_name_for_template:
                        template_filename = f"{current_class_name_for_template}_template.pcd"
                        save_path = os.path.join(PCD_CLASS_TEMPLATES_DIR, template_filename)
                        try:
                            # Ensure PCD_CLASS_TEMPLATES_DIR exists (already done at start, but good practice)
                            os.makedirs(PCD_CLASS_TEMPLATES_DIR, exist_ok=True)
                            
                            print(f"正在处理模板 '{current_class_name_for_template}'...")
                            print(f"  模板点云包含颜色: {current_object_pcd_for_template.has_colors()}")
                            if current_object_pcd_for_template.has_colors():
                                print(f"    前3个颜色值: {np.asarray(current_object_pcd_for_template.colors)[:3]}")
                            
                            # === 原点处理 ===
                            print("\n--- 开始原点处理 ---")
                            origin_mode = TEMPLATE_ORIGIN_MODE
                            if origin_mode == "interactive":
                                origin_mode = interactive_origin_selection()
                            
                            # 显示处理前的点云信息
                            points_before = np.asarray(current_object_pcd_for_template.points)
                            print(f"  处理前点云范围:")
                            print(f"    X: [{np.min(points_before[:, 0]):.3f}, {np.max(points_before[:, 0]):.3f}]")
                            print(f"    Y: [{np.min(points_before[:, 1]):.3f}, {np.max(points_before[:, 1]):.3f}]")
                            print(f"    Z: [{np.min(points_before[:, 2]):.3f}, {np.max(points_before[:, 2]):.3f}]")
                            
                            # 处理原点
                            processed_pcd = process_template_origin(current_object_pcd_for_template, origin_mode)
                            
                            # 显示处理后的点云信息  
                            points_after = np.asarray(processed_pcd.points)
                            print(f"  处理后点云范围:")
                            print(f"    X: [{np.min(points_after[:, 0]):.3f}, {np.max(points_after[:, 0]):.3f}]")
                            print(f"    Y: [{np.min(points_after[:, 1]):.3f}, {np.max(points_after[:, 1]):.3f}]")
                            print(f"    Z: [{np.min(points_after[:, 2]):.3f}, {np.max(points_after[:, 2]):.3f}]")
                            print("--- 原点处理完成 ---\n")
                            
                            # 可视化处理后的点云
                            print("显示处理后的模板点云...")
                            o3d.visualization.draw_geometries([processed_pcd], window_name=f"处理后: {current_class_name_for_template} 模板")
                            
                            # 保存处理后的点云
                            print(f"保存模板到 '{save_path}'...")
                            o3d.io.write_point_cloud(save_path, processed_pcd, write_ascii=False)
                            print(f"  成功: 模板已保存到 {save_path}")
                            
                            current_object_pcd_for_template = None # Clear after saving
                            current_class_name_for_template = None
                        except Exception as e_save:
                            print(f"  错误: 保存模板失败 '{save_path}': {e_save}")
                    else:
                        print("没有准备好的对象点云可保存。请先按 'd' 检测并准备对象。")
                
                # Always draw current capture status on display_image if object is pending save
                if current_class_name_for_template and current_object_pcd_for_template:
                     cv2.putText(display_image, f"READY TO SAVE: {current_class_name_for_template}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


                cv2.imshow("Create PCD Template (Orbbec)", display_image)

        except KeyboardInterrupt:
            print("Ctrl+C detected. Exiting template creator...")
        finally:
            if camera_instance:
                print("Stopping Orbbec camera...")
                camera_instance.stop()
            cv2.destroyAllWindows()
            print("Create PCD Template script finished.")

    except Exception as e: # Outer try-except for major init failures
        print(f"FATAL ERROR in main_create_template: {e}")
        import traceback
        traceback.print_exc()
        if camera_instance: camera_instance.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(CONFIG_YAML_FILE):
        # Try to find it relative to script dir if not in CWD
        script_dir_conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_YAML_FILE)
        if os.path.exists(script_dir_conf):
            CONFIG_YAML_FILE = script_dir_conf
        else:
            print(f"ERROR: Main config file '{CONFIG_YAML_FILE}' not found in CWD or script directory."); sys.exit(1)
            
    if not os.path.exists(YOLO_MODEL_PATH):
        script_dir_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), YOLO_MODEL_PATH)
        if os.path.exists(script_dir_model):
            YOLO_MODEL_PATH = script_dir_model
        else:
            print(f"ERROR: YOLO model file '{YOLO_MODEL_PATH}' not found in CWD or script directory."); sys.exit(1)

    if not YOLO_TARGET_CLASS_NAMES:
        print("ERROR: YOLO_TARGET_CLASS_NAMES list is empty. Please specify target classes."); sys.exit(1)

    main_create_template() 