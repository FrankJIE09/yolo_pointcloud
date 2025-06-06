import cv2
import numpy as np
import open3d as o3d
import torch
from ultralytics import YOLO
import os
import sys
import time
import yaml
import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import subprocess
import re
import json

# 假设OrbbecCamera和get_serial_numbers已在camera/orbbec_camera.py中实现
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers
except ImportError as e:
    print(f"ERROR: Failed to import OrbbecCamera module: {e}")
    sys.exit(1)

# ==== 模板类别到模型路径映射（请根据实际情况修改）====
CLASS_TO_MODEL_FILE_MAP = {
    'truck': 'lixun.STL',
    'remote': 'lixun.STL',
    'boat': 'lixun.STL',
    # 其他类别...
}
PCD_CLASS_TEMPLATES_DIR = "./pcd_class_templates"
USE_PCD_TEMPLATE_IF_AVAILABLE = True
PART2_SCRIPT_NAME = "_estimate_pose_part2_icp_estimation.py"
CONFIG_YAML_FILE = "pose_estimation_config.yaml"
MODEL_SAMPLE_POINTS_FOR_PART2 = 2048 * 5  # 添加缺失的常量


# --- 准备Part2中间数据函数（修正版本）---
def prepare_intermediate_data_for_part2(base_dir, object_pcd_o3d, model_input_path, is_pcd_template,
                                        orchestrator_args_dict, full_scene_pcd_o3d):
    """
    为Part2准备中间数据
    """
    import copy

    os.makedirs(base_dir, exist_ok=True)
    print(f"Preparing intermediate data for Part 2 in: {base_dir}")

    # 1. Save args.json
    args_file_path = os.path.join(base_dir, "args.json")
    try:
        with open(args_file_path, 'w') as f:
            json.dump(orchestrator_args_dict, f, indent=4)
        print(f"  Saved orchestrator args to {args_file_path}")
    except Exception as e:
        print(f"  ERROR saving args.json: {e}")
        return None

    # 2. Save scene_observed_for_icp.pcd (the cropped object)
    path_scene_observed_for_icp = os.path.join(base_dir, "scene_observed_for_icp.pcd")
    try:
        o3d.io.write_point_cloud(path_scene_observed_for_icp, object_pcd_o3d)
        print(f"  Saved observed object (for ICP source) to: {path_scene_observed_for_icp}")
    except Exception as e:
        print(f"  ERROR saving scene_observed_for_icp.pcd: {e}")
        return None

    # 2b. Save as instance_0_preprocessed.pcd
    path_instance_0_preprocessed = os.path.join(base_dir, "instance_0_preprocessed.pcd")
    try:
        o3d.io.write_point_cloud(path_instance_0_preprocessed, object_pcd_o3d)
        print(f"  Saved observed object also as instance_0_preprocessed.pcd to: {path_instance_0_preprocessed}")
    except Exception as e:
        print(f"  ERROR saving instance_0_preprocessed.pcd: {e}")
        return None

    # 2c. Save instance_0_centroid.npy
    path_instance_0_centroid = os.path.join(base_dir, "instance_0_centroid.npy")
    try:
        if object_pcd_o3d.has_points():
            instance_0_centroid_np = object_pcd_o3d.get_center()
            np.save(path_instance_0_centroid, instance_0_centroid_np)
            print(f"  Saved instance_0_centroid.npy to: {path_instance_0_centroid}")
        else:
            print("  Warning: object_pcd_o3d is empty, cannot save instance_0_centroid.npy. Part 2 might fail.")
    except Exception as e:
        print(f"  ERROR saving instance_0_centroid.npy: {e}")
        return None

    # 3. 加载原始CAD/PCD，计算质心
    try:
        if not model_input_path or not os.path.exists(model_input_path):
            print(f"  ERROR: Target model input file not found or not specified: {model_input_path}")
            return None

        if is_pcd_template:
            print(f"  Processing PCD template: {model_input_path}")
            target_pcd_original = o3d.io.read_point_cloud(model_input_path)
            if not target_pcd_original.has_points():
                raise ValueError(f"PCD template {model_input_path} is empty.")
        else:
            print(f"  Processing CAD model: {model_input_path}")
            temp_mesh = o3d.io.read_triangle_mesh(model_input_path)
            if not temp_mesh.has_vertices():
                print(f"  Warning: Could not read {model_input_path} as mesh. Attempting to read as point cloud.")
                target_pcd_original = o3d.io.read_point_cloud(model_input_path)
                if not target_pcd_original.has_points():
                    raise ValueError(
                        f"CAD model {model_input_path} could not be read as mesh or point cloud, or is empty.")
            else:
                # 采样出高密度点云作为原始点云
                target_pcd_original = temp_mesh.sample_points_uniformly(MODEL_SAMPLE_POINTS_FOR_PART2)

        # 只用原始点云的质心
        target_centroid_original_np = target_pcd_original.get_center()

        # 保存原始点云
        path_common_target_orig_scale = os.path.join(base_dir, "common_target_model_original_scale.pcd")
        o3d.io.write_point_cloud(path_common_target_orig_scale, target_pcd_original)

        # 保存质心
        path_common_target_centroid = os.path.join(base_dir, "common_target_centroid_original_model_scale.npy")
        np.save(path_common_target_centroid, target_centroid_original_np)

        # 保存模型路径
        path_model_file_txt = os.path.join(base_dir, "model_file_path.txt")
        with open(path_model_file_txt, 'w') as f_model:
            f_model.write(model_input_path)

        # 采样得到ICP用点云
        if len(target_pcd_original.points) >= MODEL_SAMPLE_POINTS_FOR_PART2:
            target_pcd_for_icp = target_pcd_original.farthest_point_down_sample(MODEL_SAMPLE_POINTS_FOR_PART2)
        else:
            target_pcd_for_icp = copy.deepcopy(target_pcd_original)

        # 用原始质心居中
        target_pcd_centered = copy.deepcopy(target_pcd_for_icp)
        target_pcd_centered.translate(-target_centroid_original_np)

        path_target_centered_for_icp = os.path.join(base_dir, "common_target_model_centered.pcd")
        o3d.io.write_point_cloud(path_target_centered_for_icp, target_pcd_centered)

        print(f"  Saved target model files (original, centroid, centered for ICP) to {base_dir}")

    except Exception as e_load_target:
        print(f"  ERROR processing target model from '{model_input_path}': {e_load_target}")
        return None

    # 4. Save initial_transform_for_icp.npy (identity matrix as Part 1 is skipped)
    path_instance_0_pca_transform = os.path.join(base_dir, "instance_0_pca_transform.npy")
    try:
        initial_transform_np = np.identity(4)
        np.save(path_instance_0_pca_transform, initial_transform_np)
        print(f"  Saved identity initial transform as instance_0_pca_transform.npy to: {path_instance_0_pca_transform}")
    except Exception as e_save_transform:
        print(f"  ERROR saving instance_0_pca_transform.npy: {e_save_transform}")
        return None

    # 5. Save common_original_scene.pcd (full scene point cloud)
    path_common_original_scene = os.path.join(base_dir, "common_original_scene.pcd")
    try:
        if full_scene_pcd_o3d is not None and full_scene_pcd_o3d.has_points():
            o3d.io.write_point_cloud(path_common_original_scene, full_scene_pcd_o3d, write_ascii=False)
            print(f"  Saved full original scene to: {path_common_original_scene}")
        else:
            print("  Warning: Full scene PCD not provided or is empty. common_original_scene.pcd will not be saved.")
    except Exception as e_save_scene:
        print(f"  ERROR saving common_original_scene.pcd: {e_save_scene}")

    return base_dir


def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"配置文件 {yaml_path} 不存在，使用默认参数。")
        return {}
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def matrix_to_euler_deg(R_mat):
    # ZYX顺序，返回角度
    r = R.from_matrix(R_mat)
    euler = r.as_euler('zyx', degrees=True)
    return euler[::-1]  # 返回rx, ry, rz (roll, pitch, yaw)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def find_script_path(script_name):
    path_in_cwd = os.path.join(os.getcwd(), script_name)
    if os.path.exists(path_in_cwd) and os.path.isfile(path_in_cwd):
        return os.path.abspath(path_in_cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_in_script_dir = os.path.join(script_dir, script_name)
    if os.path.exists(path_in_script_dir) and os.path.isfile(path_in_script_dir):
        return os.path.abspath(path_in_script_dir)
    return script_name


# 主流程
if __name__ == "__main__":
    # 1. 加载YAML配置
    DEFAULT_YAML = "repeatability_test_config.yaml"
    config = load_config(DEFAULT_YAML)
    test_cfg = config.get('RepeatabilityTest', {})
    num_trials = int(test_cfg.get('num_trials', 20))
    interval_sec = float(test_cfg.get('interval_sec', 1.0))
    target_class = str(test_cfg.get('target_class', 'truck'))
    output_dir = str(test_cfg.get('output_dir', './repeatability_test_results'))
    visualize_first_n = int(test_cfg.get('visualize_first_n', 3))
    show_plot = bool(test_cfg.get('show_plot', True))
    yolo_model_path = str(test_cfg.get('yolo_model', 'yolov8m.pt'))
    yolo_conf_threshold = float(test_cfg.get('yolo_conf_threshold', 0.3))

    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, f"repeatability_{target_class}.csv")
    plot_path = os.path.join(output_dir, f"repeatability_{target_class}_plot.png")
    stat_path = os.path.join(output_dir, f"repeatability_{target_class}_stat.csv")

    # 2. 初始化YOLO和相机
    print(f"加载YOLO模型: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    print("初始化Orbbec相机...")
    sns = get_serial_numbers()
    if not sns:
        print("未检测到Orbbec设备，退出。")
        sys.exit(1)
    camera = OrbbecCamera(sns[0])
    camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
    print("相机初始化完成。")

    # 3. 主循环
    results = []
    for i in range(num_trials):
        print(f"\n===== 第{i + 1}/{num_trials}次检测 =====")
        # 采集一帧
        color_img, _, depth_frame = camera.get_frames()
        if color_img is None:
            print("采集彩色图像失败，跳过本次。")
            time.sleep(interval_sec)
            continue
        # YOLO检测
        yolo_results = yolo_model(color_img, conf=yolo_conf_threshold)
        detections = yolo_results[0] if yolo_results and len(yolo_results) > 0 else None
        if not detections or len(detections.boxes) == 0:
            print("未检测到任何目标，跳过本次。")
            time.sleep(interval_sec)
            continue
        # 查找目标类别
        found = False
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = detections.names.get(cls_id, f"ClassID_{cls_id}")
            if class_name == target_class and conf >= yolo_conf_threshold:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                found = True
                break
        if not found:
            print(f"未检测到目标类别 {target_class}，跳过本次。")
            time.sleep(interval_sec)
            continue
        # 获取点云
        raw_data = camera.get_point_cloud(colored=True)
        if raw_data is None or not isinstance(raw_data, np.ndarray) or raw_data.shape[0] == 0:
            print("点云采集失败，跳过本次。")
            time.sleep(interval_sec)
            continue
        # 转为Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_data[:, :3])
        if raw_data.shape[1] >= 6:
            colors = raw_data[:, 3:6]
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float64) / 255.0
            elif np.any(colors > 1.0):
                colors = np.clip(colors / 255.0, 0, 1)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 裁剪ROI
        xmin, ymin, xmax, ymax = xyxy
        points_np = np.asarray(pcd.points)

        # 获取相机内参
        fx = camera.param.rgb_intrinsic.fx
        fy = camera.param.rgb_intrinsic.fy
        cx = camera.param.rgb_intrinsic.cx
        cy = camera.param.rgb_intrinsic.cy

        X, Y, Z = points_np[:, 0], points_np[:, 1], points_np[:, 2]
        u = (X * fx / (Z + 1e-6)) + cx
        v = (Y * fy / (Z + 1e-6)) + cy
        mask = (u >= xmin) & (u < xmax) & (v >= ymin) & (v < ymax)
        cropped_points = points_np[mask]

        if cropped_points.shape[0] < 10:
            print("裁剪后点云过少，跳过本次。")
            time.sleep(interval_sec)
            continue

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)

        # 如果原始点云有颜色，保留裁剪后的颜色
        if pcd.has_colors():
            colors_np = np.asarray(pcd.colors)
            cropped_colors = colors_np[mask]
            cropped_pcd.colors = o3d.utility.Vector3dVector(cropped_colors)

        # --- 模板选择 ---
        target_cad_model_file_path = CLASS_TO_MODEL_FILE_MAP.get(target_class)
        model_path_for_part2 = None
        is_pcd_template = False
        pcd_template_file_path = os.path.join(PCD_CLASS_TEMPLATES_DIR, f"{target_class}_template.pcd")

        if USE_PCD_TEMPLATE_IF_AVAILABLE and os.path.exists(pcd_template_file_path):
            model_path_for_part2 = pcd_template_file_path
            is_pcd_template = True
            print(f"  INFO: 优先使用PCD模板: {model_path_for_part2}")
        elif target_cad_model_file_path and os.path.exists(target_cad_model_file_path):
            model_path_for_part2 = target_cad_model_file_path
            is_pcd_template = False
            print(f"  INFO: 使用CAD模型: {model_path_for_part2}")
        elif os.path.exists(pcd_template_file_path):
            model_path_for_part2 = pcd_template_file_path
            is_pcd_template = True
            print(f"  INFO: 使用PCD模板: {model_path_for_part2}")
        else:
            print(
                f"  ERROR: 未找到CAD模型或PCD模板: {target_cad_model_file_path}, {pcd_template_file_path}，跳过该目标。")
            time.sleep(interval_sec)
            continue
        # --- End 模板选择 ---

        # --- 准备Part2中间数据 ---
        trial_dir = os.path.join(output_dir, f"trial_{i + 1:02d}")
        intermediate_dir_for_part2 = os.path.join(trial_dir, "intermediate_for_part2")
        orchestrator_args_for_part2_json = {
            "yolo_model": yolo_model_path,
            "yolo_target_classes": [target_class],
            "invoked_by_yolo_orchestrator_repeatability_test": True
        }

        prepared_intermediate_path = prepare_intermediate_data_for_part2(
            intermediate_dir_for_part2,
            cropped_pcd,
            model_path_for_part2,
            is_pcd_template,
            orchestrator_args_for_part2_json,
            pcd  # 这里用全场景点云（可根据需要调整）
        )

        if not prepared_intermediate_path:
            print("  ERROR: Failed to prepare intermediate data for Part 2. Skipping.")
            time.sleep(interval_sec)
            continue

        # --- 调用Part2 ---
        part2_script_path = find_script_path(PART2_SCRIPT_NAME)
        config_yaml_path = find_script_path(CONFIG_YAML_FILE)
        cmd_part2 = [
            sys.executable, part2_script_path,
            "--intermediate_dir", intermediate_dir_for_part2,
            "--config_file", config_yaml_path,
            "--save_results"
        ]

        print(f"  调用Part2: {' '.join(cmd_part2)}")
        result = subprocess.run(cmd_part2, capture_output=True, text=True)
        stdout = result.stdout

        # --- 解析npz路径 ---
        m = re.search(r"Saved PyTorch ICP poses to (.*\.npz)", stdout)
        if not m:
            print("  ERROR: 未能从Part2输出中解析出npz文件路径。\n输出如下:\n" + stdout)
            time.sleep(interval_sec)
            continue

        npz_file_path = m.group(1).strip()
        if not os.path.exists(npz_file_path):
            print(f"  ERROR: npz文件不存在: {npz_file_path}")
            time.sleep(interval_sec)
            continue

        # --- 读取位姿 ---
        try:
            estimated_poses_data = np.load(npz_file_path)
            pose_mat = None
            for instance_key in estimated_poses_data.files:
                pose_mat = estimated_poses_data[instance_key]
                break  # 只取第一个
            if pose_mat is None:
                print("  ERROR: npz文件中未找到位姿矩阵。")
                time.sleep(interval_sec)
                continue
        except Exception as e:
            print(f"  ERROR: 加载npz文件失败: {e}")
            time.sleep(interval_sec)
            continue

        # === 输出本次匹配的4x4位姿矩阵 ===
        print(f"本次检测的匹配结果矩阵T (第{i + 1}次):\n{pose_mat}")
        tx, ty, tz = pose_mat[:3, 3]
        rx, ry, rz = matrix_to_euler_deg(pose_mat[:3, :3])
        results.append([i + 1, tx, ty, tz, rx, ry, rz])

        # 保存到CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if i == 0 and f.tell() == 0:
                writer.writerow(['index', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
            writer.writerow([i + 1, tx, ty, tz, rx, ry, rz])

        # 可视化前几次检测的点云
        if i < visualize_first_n:
            o3d.visualization.draw_geometries([cropped_pcd], window_name=f"第{i + 1}次检测点云")

        time.sleep(interval_sec)

    # 4. 详细统计分析
    arr = np.array(results)
    if arr.shape[0] == 0:
        print("无有效检测结果，退出。")
        camera.stop()
        sys.exit(0)

    # 提取位姿数据 (跳过第一列的索引)
    pose_data = arr[:, 1:]  # tx, ty, tz, rx, ry, rz

    # 计算各种统计指标
    mean = np.mean(pose_data, axis=0)
    std = np.std(pose_data, axis=0)
    var = np.var(pose_data, axis=0)
    min_vals = np.min(pose_data, axis=0)
    max_vals = np.max(pose_data, axis=0)
    range_vals = max_vals - min_vals
    median = np.median(pose_data, axis=0)

    # 计算重复定位精度相关指标
    # 1. 3-sigma精度 (99.7%置信区间)
    precision_3sigma = 3 * std

    # 2. 最大偏差（相对于均值）
    max_deviation = np.max(np.abs(pose_data - mean), axis=0)

    # 3. 平移重复定位精度 (3D距离标准差)
    translation_data = pose_data[:, :3]  # tx, ty, tz
    translation_distances = np.sqrt(np.sum((translation_data - mean[:3]) ** 2, axis=1))
    translation_repeatability = np.std(translation_distances)
    translation_max_deviation_3d = np.max(translation_distances)

    # 4. 旋转重复定位精度 (3D角度标准差)
    rotation_data = pose_data[:, 3:]  # rx, ry, rz
    rotation_angles = np.sqrt(np.sum((rotation_data - mean[3:]) ** 2, axis=1))
    rotation_repeatability = np.std(rotation_angles)
    rotation_max_deviation_3d = np.max(rotation_angles)

    print("\n" + "=" * 60)
    print("                    详细统计分析结果")
    print("=" * 60)

    print(f"\n检测次数: {len(results)}")
    print(f"目标类别: {target_class}")

    # 基础统计指标
    print("\n【基础统计指标】")
    print("                    tx(mm)        ty(mm)        tz(mm)        rx(°)         ry(°)         rz(°)")
    print(
        f"均值         {mean[0]:12.6f} {mean[1]:12.6f} {mean[2]:12.6f} {mean[3]:12.6f} {mean[4]:12.6f} {mean[5]:12.6f}")
    print(f"标准差       {std[0]:12.6f} {std[1]:12.6f} {std[2]:12.6f} {std[3]:12.6f} {std[4]:12.6f} {std[5]:12.6f}")
    print(f"方差         {var[0]:12.6f} {var[1]:12.6f} {var[2]:12.6f} {var[3]:12.6f} {var[4]:12.6f} {var[5]:12.6f}")
    print(
        f"中位数       {median[0]:12.6f} {median[1]:12.6f} {median[2]:12.6f} {median[3]:12.6f} {median[4]:12.6f} {median[5]:12.6f}")
    print(
        f"最小值       {min_vals[0]:12.6f} {min_vals[1]:12.6f} {min_vals[2]:12.6f} {min_vals[3]:12.6f} {min_vals[4]:12.6f} {min_vals[5]:12.6f}")
    print(
        f"最大值       {max_vals[0]:12.6f} {max_vals[1]:12.6f} {max_vals[2]:12.6f} {max_vals[3]:12.6f} {max_vals[4]:12.6f} {max_vals[5]:12.6f}")
    print(
        f"范围         {range_vals[0]:12.6f} {range_vals[1]:12.6f} {range_vals[2]:12.6f} {range_vals[3]:12.6f} {range_vals[4]:12.6f} {range_vals[5]:12.6f}")
    print(
        f"最大偏差     {max_deviation[0]:12.6f} {max_deviation[1]:12.6f} {max_deviation[2]:12.6f} {max_deviation[3]:12.6f} {max_deviation[4]:12.6f} {max_deviation[5]:12.6f}")

    # 重复定位精度指标
    print("\n【重复定位精度指标】")
    print(f"平移重复定位精度 (1σ):    {translation_repeatability:.6f} mm")
    print(f"平移重复定位精度 (3σ):    {translation_repeatability * 3:.6f} mm")
    print(f"平移最大3D偏差:           {translation_max_deviation_3d:.6f} mm")
    print(f"旋转重复定位精度 (1σ):    {rotation_repeatability:.6f}°")
    print(f"旋转重复定位精度 (3σ):    {rotation_repeatability * 3:.6f}°")
    print(f"旋转最大3D偏差:           {rotation_max_deviation_3d:.6f}°")

    # 各轴3-sigma精度
    print("\n【各轴3-sigma精度 (99.7%置信区间)】")
    print(f"tx (3σ): ±{precision_3sigma[0]:.6f} mm")
    print(f"ty (3σ): ±{precision_3sigma[1]:.6f} mm")
    print(f"tz (3σ): ±{precision_3sigma[2]:.6f} mm")
    print(f"rx (3σ): ±{precision_3sigma[3]:.6f}°")
    print(f"ry (3σ): ±{precision_3sigma[4]:.6f}°")
    print(f"rz (3σ): ±{precision_3sigma[5]:.6f}°")

    # 评估等级
    print("\n【重复定位精度评估】")
    if translation_repeatability < 1.0:
        trans_grade = "优秀 (< 1mm)"
    elif translation_repeatability < 3.0:
        trans_grade = "良好 (1-3mm)"
    elif translation_repeatability < 5.0:
        trans_grade = "一般 (3-5mm)"
    else:
        trans_grade = "较差 (> 5mm)"

    if rotation_repeatability < 1.0:
        rot_grade = "优秀 (< 1°)"
    elif rotation_repeatability < 3.0:
        rot_grade = "良好 (1-3°)"
    elif rotation_repeatability < 5.0:
        rot_grade = "一般 (3-5°)"
    else:
        rot_grade = "较差 (> 5°)"

    print(f"平移精度等级: {trans_grade}")
    print(f"旋转精度等级: {rot_grade}")
    print("=" * 60)

    # 保存详细统计到csv
    with open(stat_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['统计指标', 'tx(mm)', 'ty(mm)', 'tz(mm)', 'rx(°)', 'ry(°)', 'rz(°)'])
        writer.writerow(['均值'] + mean.tolist())
        writer.writerow(['标准差'] + std.tolist())
        writer.writerow(['方差'] + var.tolist())
        writer.writerow(['中位数'] + median.tolist())
        writer.writerow(['最小值'] + min_vals.tolist())
        writer.writerow(['最大值'] + max_vals.tolist())
        writer.writerow(['范围'] + range_vals.tolist())
        writer.writerow(['最大偏差'] + max_deviation.tolist())
        writer.writerow(['3σ精度'] + precision_3sigma.tolist())
        writer.writerow([])
        writer.writerow(['重复定位精度指标', '数值', '单位'])
        writer.writerow(['平移重复定位精度(1σ)', f'{translation_repeatability:.6f}', 'mm'])
        writer.writerow(['平移重复定位精度(3σ)', f'{translation_repeatability * 3:.6f}', 'mm'])
        writer.writerow(['平移最大3D偏差', f'{translation_max_deviation_3d:.6f}', 'mm'])
        writer.writerow(['旋转重复定位精度(1σ)', f'{rotation_repeatability:.6f}', '°'])
        writer.writerow(['旋转重复定位精度(3σ)', f'{rotation_repeatability * 3:.6f}', '°'])
        writer.writerow(['旋转最大3D偏差', f'{rotation_max_deviation_3d:.6f}', '°'])
        writer.writerow(['平移精度等级', trans_grade, ''])
        writer.writerow(['旋转精度等级', rot_grade, ''])

    # 5. 绘图
    if show_plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(arr[:, 0], arr[:, 1], label='tx')
        axs[0].plot(arr[:, 0], arr[:, 2], label='ty')
        axs[0].plot(arr[:, 0], arr[:, 3], label='tz')
        axs[0].set_ylabel('Translation (m)')
        axs[0].legend()
        axs[0].set_title('平移分量随检测次数变化')

        axs[1].plot(arr[:, 0], arr[:, 4], label='rx')
        axs[1].plot(arr[:, 0], arr[:, 5], label='ry')
        axs[1].plot(arr[:, 0], arr[:, 6], label='rz')
        axs[1].set_ylabel('Euler Angle (deg)')
        axs[1].set_xlabel('检测序号')
        axs[1].legend()
        axs[1].set_title('欧拉角分量随检测次数变化')

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"结果折线图已保存: {plot_path}")
        plt.show()

    print(f"所有检测结果已保存到: {csv_path}\n统计结果: {stat_path}")
    camera.stop()
