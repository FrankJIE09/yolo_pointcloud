# -*- coding: utf-8 -*-
import time
from pyorbbecsdk import * # 导入 Orbbec SDK 的 Python 封装库
import numpy as np         # 导入 NumPy 库，用于高效的数组和矩阵操作
import yaml                # 导入 PyYAML 库，用于读写 YAML 配置文件
import json                # 导入 JSON 库 (虽然在此代码中未直接使用，但可能用于其他配置)
import cv2                 # 导入 OpenCV 库，用于图像处理和显示

# 尝试从 camera.utils 导入 frame_to_bgr_image 函数
# 这个 utils 文件需要存在于 camera 目录下，并且包含这个函数
# 该函数可能用于将 Orbbec 的 Frame 对象转换为 OpenCV 的 BGR 图像格式
try:
    from camera.utils import frame_to_bgr_image
except ImportError:
    print("警告: 无法从 'camera.utils' 导入 'frame_to_bgr_image'。")
    print("请确保 'camera/utils.py' 文件存在且包含该函数。")
    # 定义一个虚拟函数，以避免在 OrbbecCamera 类初始化时出错（如果类内部确实调用了它）
    # 注意：如果 get_point_cloud 等核心功能不依赖它，这可能不是问题。
    def frame_to_bgr_image(frame):
        print("错误: frame_to_bgr_image 函数未定义!")
        return None

ESC_KEY = 27  # 定义退出程序的按键：ESC 键的 ASCII 码

# 定义深度过滤的最小和最大阈值 (单位：毫米 mm)
# 注意: Orbbec SDK 返回的深度值或点云 Z 坐标通常是以米 (m) 为单位的浮点数。
# 在应用这些阈值时需要注意单位转换。
MIN_DEPTH = 0    # 最小深度阈值 (20mm)
MAX_DEPTH = 1000*1000  # 最大深度阈值 (1000mm = 1米) (已从原40修改)


class TemporalFilter:
    """
    一个简单的时域滤波器，用于平滑连续的帧（通常是深度图）。
    它通过将当前帧与上一帧进行加权平均来实现平滑效果。
    """
    def __init__(self, alpha=0.5):
        """
        初始化滤波器。
        Args:
            alpha (float): 当前帧的权重因子 (0 到 1 之间)。上一帧的权重是 (1 - alpha)。
                           alpha 越小，平滑效果越强，但响应越慢。
        """
        self.alpha = alpha          # 当前帧权重
        self.previous_frame = None  # 用于存储上一帧的图像数据

    def process(self, frame):
        """
        处理（滤波）输入的帧。
        Args:
            frame (np.ndarray): 当前需要滤波的帧 (通常是深度图)。
        Returns:
            np.ndarray: 滤波后的帧。
        """
        # 如果这是第一帧，直接返回它，并将其副本存储为上一帧
        if self.previous_frame is None:
            self.previous_frame = frame.copy() # 使用 copy 避免修改原始引用
            return frame

        # 在进行加权平均前，确保当前帧和上一帧的数据类型 (dtype) 相同
        if frame.dtype != self.previous_frame.dtype:
            # 尝试将上一帧转换为当前帧的数据类型
            try:
                self.previous_frame = self.previous_frame.astype(frame.dtype)
            except Exception as e:
                print(
                    f"TemporalFilter 错误: 无法将上一帧的数据类型 {self.previous_frame.dtype} 转换为 {frame.dtype}。"
                    f"正在重置滤波器。错误: {e}")
                self.previous_frame = frame.copy() # 重置滤波器
                return frame # 返回当前帧

        # 确保当前帧和上一帧的形状 (shape) 相同
        if frame.shape != self.previous_frame.shape:
            print(
                f"TemporalFilter 警告: 帧形状发生变化 ({self.previous_frame.shape} -> {frame.shape})。"
                f"正在重置滤波器。")
            self.previous_frame = frame.copy() # 重置滤波器
            return frame # 返回当前帧

        # 使用 OpenCV 的 addWeighted 函数进行加权平均
        try:
            # result = current_frame * alpha + previous_frame * (1 - alpha) + 0
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
            self.previous_frame = result.copy() # 存储当前结果作为下一次处理的上一帧
            return result
        except cv2.error as e:
            print(f"TemporalFilter 错误 (cv2.addWeighted): {e}. 正在重置滤波器。")
            # 处理潜在的数据类型/通道问题（尽管已经检查过）
            self.previous_frame = frame.copy()
            return frame


class OrbbecCamera:
    """
    封装了与 Orbbec 相机交互的主要逻辑，包括设备连接、流配置、帧获取、点云生成、IMU读取等。
    """
    def __init__(self, device_id, config_extrinsic='./hand_eye_config.yaml'):
        """
        初始化 OrbbecCamera 类。
        Args:
            device_id (str): 要连接的 Orbbec 相机的序列号 (Serial Number)。
            config_extrinsic (str): (可选) 包含手眼标定外参矩阵的 YAML 配置文件的路径。
        Raises:
            RuntimeError: 如果找不到 Orbbec 设备或指定的设备。
        """
        # 获取设备列表
        self.context = Context()
        device_list = self.context.query_devices()
        if device_list.get_count() == 0:
            raise RuntimeError(f"未找到任何 Orbbec 设备。")
        try:
            self.device = device_list.get_device_by_serial_number(device_id)
            if self.device is None:
                raise RuntimeError(f"未找到序列号为 {device_id} 的设备。")
        except OBError as e:
            raise RuntimeError(f"通过序列号 {device_id} 获取设备失败: {e}")
        except Exception as e:
            raise RuntimeError(f"获取设备 {device_id} 时发生意外错误: {e}")

        self.sensor_list = self.device.get_sensor_list()
        self.device_info = self.device.get_device_info()
        self.config_path = config_extrinsic
        self.temporal_filter = TemporalFilter()
        self.config = Config()
        self.pipeline = Pipeline(self.device)
        self.point_cloud_filter = None  # 初始化点云滤波器占位符
        self.param = None  # 初始化相机参数占位符

        # --- 获取 Depth 和 Color 流配置 ---
        try:
            profile_list_depth = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if profile_list_depth is None: raise RuntimeError("获取深度流配置列表失败。")
            self.depth_profile = profile_list_depth.get_default_video_stream_profile()
            if self.depth_profile is None: raise RuntimeError("获取默认深度视频流配置失败。")
            print(f"默认深度流配置: {self.depth_profile.get_width()}x{self.depth_profile.get_height()}@{self.depth_profile.get_fps()}fps {self.depth_profile.get_format()}")
        except OBError as e: raise RuntimeError(f"获取深度流配置时出错: {e}")

        try:
            profile_list_color = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list_color is None:
                print("警告: 获取彩色流配置列表失败。")
                self.color_profile = None
            else:
                self.color_profile = profile_list_color.get_default_video_stream_profile()
                if self.color_profile is None: print("警告: 获取默认彩色视频流配置失败。")
                else: print(f"默认彩色流配置: {self.color_profile.get_width()}x{self.color_profile.get_height()}@{self.color_profile.get_fps()}fps {self.color_profile.get_format()}")
        except OBError as e:
            print(f"警告: 获取彩色流配置时出错: {e}")
            self.color_profile = None

        # ++++++++++++++++ IMU 初始化 (已修正 AttributeError) ++++++++++++++++
        self.has_accel = False  # 标记是否存在加速度计
        self.accel_profile = None # 加速度计流配置
        self.has_gyro = False   # 标记是否存在陀螺仪
        self.gyro_profile = None  # 陀螺仪流配置

        # 尝试获取加速度计传感器和配置
        try:
            accel_sensor = self.sensor_list.get_sensor_by_type(OBSensorType.ACCEL_SENSOR)
            if accel_sensor:
                self.has_accel = True
                accel_profile_list = accel_sensor.get_stream_profile_list()
                if accel_profile_list and accel_profile_list.get_count() > 0:
                    self.accel_profile = accel_profile_list.get_stream_profile_by_index(0)
                    # 修正: 移除了 .get_gyro_accel_sample_rate() 调用, 只尝试获取量程
                    try:
                        accel_range = self.accel_profile.get_accel_full_scale_range()
                        print(f"找到加速度计传感器并获取配置: Range={accel_range}")
                    except OBError as e_range:
                        print(f"找到加速度计传感器, 但获取量程失败: {e_range}")
                    except AttributeError:
                         print(f"找到加速度计传感器, 但 get_accel_full_scale_range() 方法不存在。")
                else:
                    print("警告: 找到加速度计传感器，但无法获取流配置。")
                    self.has_accel = False
            else:
                print("此设备没有加速度计传感器。")
        except OBError as e:
            print(f"查找加速度计传感器时出错: {e}")

        # 尝试获取陀螺仪传感器和配置
        try:
            gyro_sensor = self.sensor_list.get_sensor_by_type(OBSensorType.GYRO_SENSOR)
            if gyro_sensor:
                self.has_gyro = True
                gyro_profile_list = gyro_sensor.get_stream_profile_list()
                if gyro_profile_list and gyro_profile_list.get_count() > 0:
                    self.gyro_profile = gyro_profile_list.get_stream_profile_by_index(0)
                    # 修正: 移除了 .get_gyro_accel_sample_rate() 调用, 只尝试获取量程
                    try:
                        gyro_range = self.gyro_profile.get_gyro_full_scale_range()
                        print(f"找到陀螺仪传感器并获取配置: Range={gyro_range}")
                    except OBError as e_range:
                         print(f"找到陀螺仪传感器, 但获取量程失败: {e_range}")
                    except AttributeError:
                         print(f"找到陀螺仪传感器, 但 get_gyro_full_scale_range() 方法不存在。")
                else:
                    print("警告: 找到陀螺仪传感器，但无法获取流配置。")
                    self.has_gyro = False
            else:
                print("此设备没有陀螺仪传感器。")
        except OBError as e:
            print(f"查找陀螺仪传感器时出错: {e}")
        # ++++++++++++++++ 结束 IMU 初始化 ++++++++++++++++

        self.extrinsic_matrix = []
        self.depth_fx = self.depth_fy = self.depth_ppx = self.depth_ppy = None
        self.depth_distortion = None
        self.rgb_fx = self.rgb_fy = self.rgb_ppx = self.rgb_ppy = None
        self.rgb_distortion = None
        self.stream = False

    def color_viewer(self):
        """启动一个简单的窗口，实时显示彩色摄像头的画面。"""
        if not self.stream:
            print("流尚未启动。请先调用 start_stream()。")
            try: self.start_stream()
            except Exception as e: print(f"为 color_viewer 自动启动流失败: {e}"); return

        print("按 'q' 或 ESC 键退出彩色图像查看器。")
        while True:
            try:
                frames: FrameSet = self.pipeline.wait_for_frames(100)
                if frames is None: continue
                color_frame = frames.get_color_frame()
                if color_frame is None: continue
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None: print("将彩色帧转换为图像失败。"); continue
                cv2.imshow("Color Viewer", color_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY: print("退出 Color Viewer。"); break
            except KeyboardInterrupt: print("检测到 KeyboardInterrupt，退出 Color Viewer。"); break
            except Exception as e: print(f"color_viewer 循环中出错: {e}"); break
        cv2.destroyWindow("Color Viewer")

    def get_device_name(self):
        """返回设备名称。"""
        return self.device_info.get_name()

    def get_device_pid(self):
        """返回设备的产品 ID (PID)。"""
        return self.device_info.get_pid()

    def get_serial_number(self):
        """返回设备的序列号。"""
        return self.device_info.get_serial_number()

    def set_auto_exposure(self, auto_exposure: bool):
        """设置彩色摄像头的自动曝光开关。"""
        try:
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, auto_exposure)
            print(f"彩色摄像头自动曝光设置为: {auto_exposure}")
        except OBError as e:
            print(f"设置自动曝光失败: {e}")

    def get_current_exposure(self):
        """获取彩色摄像头当前的曝光值。"""
        try:
            return self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
        except OBError as e:
            print(f"获取当前曝光值失败: {e}")
            return None

    def set_exposure(self, exposure_value: int):
        """设置彩色摄像头的曝光值。"""
        try:
            self.set_auto_exposure(False)
            time.sleep(0.1)
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_value)
            print(f"彩色摄像头曝光值设置为: {exposure_value}")
        except OBError as e:
            print(f"设置曝光值失败: {e}")

    def adjust_exposure(self, adjustment: int):
        """在当前曝光值的基础上进行调整。"""
        curr_exposure = self.get_current_exposure()
        if curr_exposure is not None:
            new_exposure = curr_exposure + adjustment
            self.set_exposure(new_exposure)

    def set_software_filter(self, soft_filter: bool):
        """开启或关闭深度数据软件滤波。"""
        try:
            self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, soft_filter)
            print(f"深度数据软件滤波设置为: {soft_filter}")
        except OBError as e:
            print(f"设置深度软件滤波失败: {e}")

    def reboot(self):
        """重启 Orbbec 设备。"""
        try:
            print("正在重启设备...")
            self.device.reboot()
            print("设备重启命令已发送。可能需要重新初始化。")
            self.stream = False # 标记流已停止
        except OBError as e:
            print(f"重启设备失败: {e}")

    def start_stream(self, depth_stream=True, color_stream=True,
                     enable_accel=True, enable_gyro=True, # <-- IMU 启用选项
                     enable_sync=True, use_alignment=True):
        '''配置并启动 Pipeline 数据流 (包括可选的 IMU)。'''
        self.config = Config()  # 每次启动前重置配置

        # 配置 Color 流
        if color_stream and self.color_profile:
            print("启用 Color 流...")
            self.config.enable_stream(self.color_profile)
        elif color_stream and not self.color_profile:
            print("警告: 请求启用 Color 流, 但未找到有效配置。")

        # 配置 Depth 流
        if depth_stream and self.depth_profile:
            print("启用 Depth 流...")
            self.config.enable_stream(self.depth_profile)
            if use_alignment and color_stream and self.color_profile:
                print("设置对齐模式为 SW_MODE (Depth to Color)。")
                self.config.set_align_mode(OBAlignMode.SW_MODE)
            elif use_alignment:
                print("警告: 请求启用对齐, 但 Color 流被禁用或不可用。")
        elif depth_stream and not self.depth_profile:
            print("错误: 请求启用 Depth 流, 但未找到有效配置。")

        # 配置 Accel 流
        if enable_accel and self.has_accel and self.accel_profile:
            print("启用 Accel (加速度计) 流...")
            self.config.enable_stream(self.accel_profile)
        elif enable_accel:
            print("警告: 请求启用 Accel 流, 但传感器不存在或配置无效。")

        # 配置 Gyro 流
        if enable_gyro and self.has_gyro and self.gyro_profile:
            print("启用 Gyro (陀螺仪) 流...")
            self.config.enable_stream(self.gyro_profile)
        elif enable_gyro:
            print("警告: 请求启用 Gyro 流, 但传感器不存在或配置无效。")

        # 检查是否有任何有效的流被启用 (视觉流优先)
        has_visual_stream = (depth_stream and self.depth_profile) or (color_stream and self.color_profile)
        has_imu_stream = (enable_accel and self.has_accel and self.accel_profile) or \
                         (enable_gyro and self.has_gyro and self.gyro_profile)

        if not has_visual_stream and not has_imu_stream:
            raise RuntimeError("没有选择或没有可用的有效数据流来启动。")
        elif not has_visual_stream and has_imu_stream:
            print("警告：仅启动 IMU 流。")

        print("正在启动 Pipeline...")
        try:
            self.pipeline.start(self.config)
        except OBError as e:
            raise RuntimeError(f"启动 Pipeline 失败: {e}")

        self.stream = True
        print("Pipeline 已启动。")

        # 尝试启用帧同步
        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
                print("帧同步已启用。")
            except OBError as e: print(f"警告: 启用帧同步失败: {e}")
            except Exception as e: print(f"警告: 启用帧同步时发生意外错误: {e}")

        # 获取相机参数
        try:
            self.param = self.pipeline.get_camera_param()
            print("相机参数已获取。")
            self.depth_fx = self.param.depth_intrinsic.fx
            self.depth_fy = self.param.depth_intrinsic.fy
            self.depth_ppx = self.param.depth_intrinsic.cx
            self.depth_ppy = self.param.depth_intrinsic.cy
            self.depth_distortion = self.param.depth_distortion
            if self.color_profile:
                self.rgb_fx = self.param.rgb_intrinsic.fx
                self.rgb_fy = self.param.rgb_intrinsic.fy
                self.rgb_ppx = self.param.rgb_intrinsic.cx
                self.rgb_ppy = self.param.rgb_intrinsic.cy
                self.rgb_distortion = self.param.rgb_distortion
        except OBError as e:
            print(f"警告: 获取相机参数失败: {e}")
            self.param = None

        # 初始化点云滤波器
        if self.param is not None:
            try:
                self.point_cloud_filter = PointCloudFilter()
                self.point_cloud_filter.set_camera_param(self.param)
                print("PointCloudFilter 已创建并配置。")
            except OBError as e:
                print(f"创建 PointCloudFilter 时出错: {e}")
                self.point_cloud_filter = None
        else:
            print("警告: 由于缺少相机参数，无法初始化 PointCloudFilter。")
            self.point_cloud_filter = None

    def get_frames(self):
        """
        等待并获取一帧数据集合(FrameSet)，从中提取处理后的彩色图像、深度数据、
        原始深度帧以及 IMU 数据 (如果已启用)。
        (已修正 get_frame 调用参数类型)

        Returns:
            tuple: (color_image, depth_data, depth_frame, accel_data, gyro_data)
                   color_image (np.ndarray | None): BGR 彩色图像。
                   depth_data (np.ndarray | None): uint16 毫米单位深度数据。
                   depth_frame (DepthFrame | None): 原始(可能已对齐)深度帧对象。
                   accel_data (dict | None): 加速度计数据 {'ts': float, 'x': float, 'y': float, 'z': float} 或 None。
                   gyro_data (dict | None): 陀螺仪数据 {'ts': float, 'x': float, 'y': float, 'z': float} 或 None。
        """
        if not self.stream:
            print("流尚未启动。请先调用 start_stream()。")
            return None, None, None, None, None  # 返回值数量匹配

        frames: FrameSet = None
        try:
            frames = self.pipeline.wait_for_frames(100)
        except OBError as e:
            print(f"等待帧时出错: {e}")
            return None, None, None, None, None

        if frames is None:
            return None, None, None, None, None

        # 获取 Depth 和 Color 帧 (逻辑不变)
        depth_frame: DepthFrame = frames.get_depth_frame()
        color_frame: ColorFrame = frames.get_color_frame()

        # 处理 Depth 和 Color (逻辑不变)
        color_image = self.color_frame2color_image(color_frame) if color_frame else None
        depth_data = self.depth_frame2depth_data(depth_frame, filter_on=True) if depth_frame else None
        print(depth_data)
        if depth_data is not None:
            print(depth_data.max())
        # ++++++++++++++++ 获取和处理 IMU 数据 (已修正参数类型) ++++++++++++++++
        accel_data = None
        gyro_data = None

        # 尝试获取 Accel 帧
        if self.has_accel:
            # 1. 从 FrameSet 获取通用的 Frame 对象, 使用 OBFrameType
            generic_accel_frame: Frame = frames.get_frame(OBFrameType.ACCEL_FRAME)  # <-- 修改处
            if generic_accel_frame:
                # 2. 将通用 Frame 对象转换为 AccelFrame 对象
                accel_frame: AccelFrame = generic_accel_frame.as_accel_frame()
                if accel_frame:
                    # 3. 提取数据
                    accel_data = {
                        'ts': accel_frame.get_timestamp(),
                        'x': accel_frame.get_x(),
                        'y': accel_frame.get_y(),
                        'z': accel_frame.get_z()
                    }

        # 尝试获取 Gyro 帧
        if self.has_gyro:
            # 1. 从 FrameSet 获取通用的 Frame 对象, 使用 OBFrameType
            generic_gyro_frame: Frame = frames.get_frame(OBFrameType.GYRO_FRAME)  # <-- 修改处
            if generic_gyro_frame:
                # 2. 将通用 Frame 对象转换为 GyroFrame 对象
                gyro_frame: GyroFrame = generic_gyro_frame.as_gyro_frame()
                if gyro_frame:
                    # 3. 提取数据
                    gyro_data = {
                        'ts': gyro_frame.get_timestamp(),
                        'x': gyro_frame.get_x(),
                        'y': gyro_frame.get_y(),
                        'z': gyro_frame.get_z()
                    }
        # ++++++++++++++++ 结束 IMU 数据处理 ++++++++++++++++

        # 返回所有数据
        return color_image, depth_data, depth_frame, accel_data, gyro_data
    def is_streaming(self):
        """检查流是否正在运行 (基于内部标志)。"""
        return self.stream

    def get_point_cloud(self, colored=True):
        """
        等待帧数据，计算点云，并根据深度范围进行过滤，然后返回点云数据。

        Args:
            colored (bool): 是否生成彩色点云。

        Returns:
            np.ndarray | None: 过滤后的点云 NumPy 数组 (Nx3 或 Nx6) 或 None。
        """
        if not self.stream: print("get_point_cloud 错误: 流尚未启动。"); return None
        if self.point_cloud_filter is None: print("get_point_cloud 错误: PointCloudFilter 未初始化。"); return None

        frames = None
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None: return None
            depth_frame = frames.get_depth_frame()
            if depth_frame is None: return None

            color_frame = frames.get_color_frame() if colored else None
            has_valid_color_for_pc = colored and (color_frame is not None)
            alignment_enabled = False
            if hasattr(self.config, 'get_align_mode'): alignment_enabled = self.config.get_align_mode() != OBAlignMode.ALIGN_DISABLE
            if colored and not has_valid_color_for_pc: print("get_point_cloud 警告: 缺少彩色帧，回退到 XYZ。"); has_valid_color_for_pc = False
            elif colored and not alignment_enabled: print("get_point_cloud 警告: 对齐未启用，回退到 XYZ。"); has_valid_color_for_pc = False

            scale = depth_frame.get_depth_scale()
            self.point_cloud_filter.set_position_data_scaled(scale)
            point_format = OBFormat.RGB_POINT if has_valid_color_for_pc else OBFormat.POINT
            self.point_cloud_filter.set_create_point_format(point_format)

            point_cloud_frame = self.point_cloud_filter.process(frames)
            if point_cloud_frame is None: return None
            points_data = self.point_cloud_filter.calculate(point_cloud_frame)
            if points_data is None: return None

            try: points = np.array(points_data, dtype=np.float32)
            except ValueError as e: print(f"get_point_cloud 错误: 转换 NumPy 数组失败: {e}"); return None

            if points.size == 0: return None
            expected_cols = 6 if point_format == OBFormat.RGB_POINT else 3
            if points.ndim != 2 or points.shape[1] != expected_cols:
                print(f"get_point_cloud 错误: 意外形状 {points.shape}, 期望 {expected_cols} 列。")
                if points.shape[1] == 3 and expected_cols == 6: return points # 返回 XYZ
                elif points.shape[1] == 6 and expected_cols == 3: return points[:, :3] # 返回 XYZ
                else: return None

            z_coordinates = points[:, 2]
            min_depth_m = MIN_DEPTH / 1000.0
            max_depth_m = MAX_DEPTH / 1000.0
            depth_mask = (z_coordinates >= min_depth_m) & (z_coordinates <= max_depth_m)
            filtered_points = points[depth_mask]
            if filtered_points.size == 0: return None

            return filtered_points

        except OBError as e: print(f"get_point_cloud: Orbbec SDK 错误 - {e}"); return None
        except Exception as e: print(f"get_point_cloud: 意外错误 - {e}"); import traceback; traceback.print_exc(); return None


    def get_depth_for_color_pixel(self, depth_frame, color_point, show=False):
        """获取给定彩色像素坐标对应的深度值(mm), 会在邻域搜索有效值。"""
        if depth_frame is None: print("get_depth_for_color_pixel 错误: 无效的 depth_frame。"); return 0
        depth_data = self.depth_frame2depth_data(depth_frame, filter_on=True)
        if depth_data is None: print("get_depth_for_color_pixel 错误: 获取 depth_data 失败。"); return 0

        if show: # Visualization logic unchanged
            import matplotlib.pyplot as plt
            depth_display = depth_data.copy()
            mask = depth_display > 0
            if np.any(mask):
                min_val, max_val = depth_display[mask].min(), depth_display[mask].max()
                if max_val > min_val: depth_display[mask] = ((depth_display[mask] - min_val) / (max_val - min_val)) * 255
                else: depth_display[mask] = 128
            depth_display = depth_display.astype(np.uint8)
            depth_image_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            depth_image_color[~mask] = [0, 0, 0]
            plt.figure(figsize=(8, 6)); plt.imshow(cv2.cvtColor(depth_image_color, cv2.COLOR_BGR2RGB)); plt.title("深度图查看器"); plt.axis("off")
            plt.scatter([color_point[0]], [color_point[1]], c='red', s=40, marker='x'); plt.show()

        center_x, center_y = int(round(color_point[0])), int(round(color_point[1]))
        if not (0 <= center_y < depth_data.shape[0] and 0 <= center_x < depth_data.shape[1]): print(f"get_depth_for_color_pixel 错误: 像素坐标 ({center_x}, {center_y}) 超出边界"); return 0
        center_depth = depth_data[center_y, center_x]

        if center_depth == 0: # Neighborhood search logic unchanged
            search_radius, max_radius, found_depth = 1, 10, 0
            while search_radius <= max_radius:
                min_x, max_x = max(center_x - search_radius, 0), min(center_x + search_radius + 1, depth_data.shape[1])
                min_y, max_y = max(center_y - search_radius, 0), min(center_y + search_radius + 1, depth_data.shape[0])
                neighborhood = depth_data[min_y:max_y, min_x:max_x]; valid_depths = neighborhood[neighborhood > 0]
                if valid_depths.size > 0: found_depth = valid_depths.min(); break
                search_radius += 1
            center_depth = found_depth
        return center_depth

    def color_frame2color_image(self, color_frame):
        """将 Orbbec 彩色帧转换为 OpenCV BGR 图像。"""
        if color_frame is None: return None
        try: return frame_to_bgr_image(color_frame)
        except Exception as e: print(f"转换彩色帧时出错: {e}"); return None

    def depth_frame2depth_data(self, depth_frame, filter_on=True):
        """处理深度帧, 返回 uint16 mm 单位数据 (应用滤波)。"""
        if depth_frame is None: return None
        try:
            h, w = depth_frame.get_height(), depth_frame.get_width(); scale = depth_frame.get_depth_scale()
            d_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            d_m = d_u16.astype(np.float32) * scale
            min_m, max_m = MIN_DEPTH / 1000.0, MAX_DEPTH / 1000.0
            d_m_filt = np.where((d_m > min_m) & (d_m < max_m), d_m, 0)
            d_mm_u16 = (d_m_filt * 1000).astype(np.uint16)
            if filter_on: d_mm_u16 = self.temporal_filter.process(d_mm_u16)
            return d_mm_u16
        except Exception as e: print(f"处理深度帧时出错: {e}"); return None

    def show_depth_frame(self, depth_frame, window_name="Depth Viewer"):
        """处理并显示深度帧 (伪彩色)。"""
        d_mm = self.depth_frame2depth_data(depth_frame, filter_on=True)
        if d_mm is None or d_mm.size == 0: return
        mask = d_mm > 0; d_disp = np.zeros_like(d_mm, dtype=np.uint8)
        if np.any(mask):
            min_v, max_v = d_mm[mask].min(), d_mm[mask].max()
            if max_v > min_v: d_disp[mask] = ((d_mm[mask] - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            elif min_v > 0: d_disp[mask] = 128
        d_color = cv2.applyColorMap(d_disp, cv2.COLORMAP_JET); d_color[~mask] = [0, 0, 0]
        cv2.imshow(window_name, d_color)

    def stop(self):
        """停止 Pipeline 数据流。"""
        if self.stream:
            print("正在停止 Pipeline...")
            try: self.pipeline.stop(); self.stream = False; print("Pipeline 已停止。")
            except OBError as e: print(f"停止 Pipeline 时出错: {e}")
        else: print("Pipeline 已经停止。")

    def adjust_exposure_based_on_brightness(self, target_brightness=100, max_iterations=10):
         """(实验性) 自动调整曝光。"""
         if not self.stream: return
         step, min_e, max_e, tol = 10, 10, 10000, 10
         print(f"调整曝光 -> 目标亮度: {target_brightness} +/- {tol}")
         for i in range(max_iterations):
             frames = self.pipeline.wait_for_frames(100);
             if frames is None: continue
             c_frame = frames.get_color_frame();
             if c_frame is None: continue
             c_img = self.color_frame2color_image(c_frame)
             if c_img is None or c_img.size == 0: continue
             curr_b = (0.299*c_img[:,:,2].mean()+0.587*c_img[:,:,1].mean()+0.114*c_img[:,:,0].mean())
             print(f"迭代 {i+1}/{max_iterations}: 当前亮度 = {curr_b:.2f}")
             curr_e = self.get_current_exposure();
             if curr_e is None: break
             if abs(curr_b - target_brightness) <= tol: break
             if curr_b < target_brightness: new_e = min(curr_e + step, max_e)
             else: new_e = max(curr_e - step, min_e)
             if new_e != curr_e: self.set_exposure(new_e); time.sleep(0.1)
             else: break
         else: print(f"调整在 {max_iterations} 次迭代后停止。")

    def load_extrinsic(self):
        """从配置文件中加载外参。"""
        try:
            with open(self.config_path, 'r') as f: config = yaml.safe_load(f)
            if config and 'hand_eye_transformation_matrix' in config:
                self.extrinsic_matrix = np.array(config['hand_eye_transformation_matrix'])
                print("外参矩阵加载成功。")
            else: print(f"错误: 未在 {self.config_path} 中找到外参键。"); self.extrinsic_matrix = np.identity(4)
        except FileNotFoundError: print(f"错误: 找不到外参文件 {self.config_path}"); self.extrinsic_matrix = np.identity(4)
        except yaml.YAMLError as e: print(f"解析 YAML {self.config_path} 出错: {e}"); self.extrinsic_matrix = np.identity(4)
        except Exception as e: print(f"加载外参时未知错误: {e}"); self.extrinsic_matrix = np.identity(4)


# === Utility functions (不变) ===

def get_serial_numbers():
    """获取所有已连接 Orbbec 设备的序列号列表。"""
    sns = []; ctx = Context()
    try:
        dev_list = ctx.query_devices(); count = dev_list.get_count(); print(f"找到 {count} 个设备。")
        for i in range(count):
            try: sn = dev_list.get_device_serial_number_by_index(i); sns.append(sn) if sn else print(f"警告: 获取索引 {i} 序列号失败。")
            except OBError as e: print(f"获取索引 {i} 序列号出错: {e}")
    except OBError as e: print(f"查询设备出错: {e}")
    except Exception as e: print(f"get_serial_numbers 意外错误: {e}")
    return sns

def initialize_connected_cameras(serial_number):
    """根据单个序列号初始化相机。"""
    try: return OrbbecCamera(serial_number)
    except Exception as e: print(f"初始化序列号 {serial_number} 失败: {e}"); return None

def initialize_all_connected_cameras(serial_numbers=None):
    """初始化指定列表或所有找到的相机。"""
    if serial_numbers is None: sns = get_serial_numbers();
    else: sns = serial_numbers if isinstance(serial_numbers, list) else [serial_numbers]
    if not sns: print("未找到/指定设备。"); return []
    devices = []; print(f"尝试初始化相机: {sns}")
    for sn in sns:
        print(f"  初始化 {sn}..."); dev = initialize_connected_cameras(sn)
        if dev: devices.append(dev); print(f"  成功初始化 {sn}。")
    return devices

def close_connected_cameras(cameras):
    """停止所有提供的相机对象的 Pipeline。"""
    if isinstance(cameras, OrbbecCamera): cameras = [cameras]
    print(f"正在关闭 {len(cameras)} 个相机...")
    for cam in cameras:
        try: cam.stop()
        except Exception as e:
            sn = "未知";
            try: sn = cam.get_serial_number()
            except: pass; print(f"停止相机 {sn} 出错: {e}")

# === Main execution block (演示获取 IMU 数据) ===
def main():
    available_sns = get_serial_numbers()
    cameras = initialize_all_connected_cameras(available_sns)
    if not cameras: print("没有相机初始化成功。退出。"); return

    active_cameras = []
    print("\n正在为所有相机启动流 (包括 IMU)...")
    for camera in cameras:
        try:
            camera.start_stream(depth_stream=True, color_stream=True,
                                enable_accel=True, enable_gyro=True, # <-- 确保启用 IMU
                                use_alignment=True, enable_sync=True)
            active_cameras.append(camera)
        except Exception as e:
            sn="未知"
            try: sn=camera.get_serial_number()
            except: pass; print(f"为相机 {sn} 启动流失败: {e}")

    if not active_cameras: print("没有相机流成功启动。退出。"); return

    print(f"\n成功启动 {len(active_cameras)} 个相机流。")
    print("进入主循环 (按 'q' 退出)...")
    try:
        while True:
            quit_flag = False
            for idx, camera in enumerate(active_cameras):
                if not camera.stream: continue

                # --- 获取所有数据帧 (包括 IMU) ---
                color_image, depth_data, depth_frame_obj, accel_data, gyro_data = camera.get_frames()

                # --- 显示图像/深度 (可选) ---
                if color_image is not None: cv2.imshow(f"Cam {idx} Color", color_image)
                if depth_frame_obj is not None: camera.show_depth_frame(depth_frame_obj, f"Cam {idx} Depth")

                # --- 打印 IMU 数据 ---
                ts_now = time.time() * 1000 # 当前系统时间 (ms) 用于比较延迟
                if accel_data:
                    delay_accel = ts_now - accel_data['ts'] if accel_data['ts'] > 0 else float('inf')
                    print(f"Cam{idx} Accel: ts={accel_data['ts']:.0f} ({delay_accel:.1f}ms ago) x={accel_data['x']:.4f}, y={accel_data['y']:.4f}, z={accel_data['z']:.4f}")
                if gyro_data:
                    delay_gyro = ts_now - gyro_data['ts'] if gyro_data['ts'] > 0 else float('inf')
                    print(f"Cam{idx} Gyro:  ts={gyro_data['ts']:.0f} ({delay_gyro:.1f}ms ago) x={gyro_data['x']:.4f}, y={gyro_data['y']:.4f}, z={gyro_data['z']:.4f}")

                # --- 获取点云 (可选) ---
                # points = camera.get_point_cloud(colored=True)
                # if points is not None: print(f"Cam {idx} Point Cloud Shape: {points.shape}")
                # (点云可视化代码...)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY: print("检测到退出按键。"); quit_flag = True; break
            if quit_flag: break

    except KeyboardInterrupt: print("\n检测到 Ctrl+C。正在退出循环。")
    finally:
        print("\n正在清理..."); close_connected_cameras(active_cameras); cv2.destroyAllWindows(); print("程序已结束。")

if __name__ == "__main__":
    main()