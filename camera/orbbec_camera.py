import time
from pyorbbecsdk import *
import numpy as np  # 导入NumPy库，用于数组和矩阵操作
import yaml
import json
import cv2
from camera.utils import frame_to_bgr_image  # Assuming frame_to_rgb_frame is not needed here

ESC_KEY = 27
MIN_DEPTH = 20  # 20mm  (Note: Pyorbbecsdk often works in meters after scaling)
MAX_DEPTH = 1000  # 1000mm (10 meters)


class TemporalFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            self.previous_frame = frame.copy()  # Use copy to avoid modifying original reference
            return frame
        # Ensure both frames have the same dtype before addWeighted
        if frame.dtype != self.previous_frame.dtype:
            # Attempt to convert previous frame to current frame's dtype
            try:
                self.previous_frame = self.previous_frame.astype(frame.dtype)
            except Exception as e:
                print(
                    f"TemporalFilter Error: Cannot cast previous frame dtype {self.previous_frame.dtype} to {frame.dtype}. Resetting filter. Error: {e}")
                self.previous_frame = frame.copy()
                return frame

        # Ensure frames have the same shape
        if frame.shape != self.previous_frame.shape:
            print(
                f"TemporalFilter Warning: Frame shape changed ({self.previous_frame.shape} -> {frame.shape}). Resetting filter.")
            self.previous_frame = frame.copy()
            return frame

        try:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
            self.previous_frame = result.copy()  # Store a copy of the result
            return result
        except cv2.error as e:
            print(f"TemporalFilter Error during cv2.addWeighted: {e}. Resetting filter.")
            # Handle potential dtype/channel issues if they arise despite checks
            self.previous_frame = frame.copy()
            return frame


class OrbbecCamera:
    def __init__(self, device_id, config_extrinsic='./hand_eye_config.yaml'):
        # 获取设备列表
        self.context = Context()
        device_list = self.context.query_devices()
        if device_list.get_count() == 0:
            raise RuntimeError(f"No Orbbec devices found.")
        try:
            self.device = device_list.get_device_by_serial_number(device_id)
            if self.device is None:
                raise RuntimeError(f"Device with serial number {device_id} not found.")
        except OBError as e:
            raise RuntimeError(f"Failed to get device by serial number {device_id}: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred getting device {device_id}: {e}")

        self.sensor_list = self.device.get_sensor_list()
        self.device_info = self.device.get_device_info()
        self.config_path = config_extrinsic
        self.temporal_filter = TemporalFilter()
        self.config = Config()
        self.pipeline = Pipeline(self.device)
        self.point_cloud_filter = None  # Initialize point cloud filter placeholder
        self.param = None  # Initialize camera params placeholder

        # --- Get Profiles (add error checking) ---
        try:
            profile_list_depth = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if profile_list_depth is None:
                raise RuntimeError("Failed to get depth stream profile list.")

            # --- Try to find a higher resolution depth profile ---
            selected_depth_profile = None
            best_profile = None
            highest_resolution = 0
            target_width = 1280  # Desired width
            target_height = 720 # Desired height

            for i in range(profile_list_depth.get_count()):
                profile = profile_list_depth.get_stream_profile_by_index(i)
                if profile:
                    video_profile = profile.as_video_stream_profile()
                    if video_profile:
                        current_resolution = video_profile.get_width() * video_profile.get_height()
                        # print(f"DEBUG: Found depth profile: {video_profile.get_width()}x{video_profile.get_height()}@{video_profile.get_fps()}fps {video_profile.get_format()}")

                        # Check for target resolution
                        if video_profile.get_width() == target_width and video_profile.get_height() == target_height:
                            # Prefer Y16 format if available at target resolution
                            if video_profile.get_format() == OBFormat.Y16:
                                selected_depth_profile = video_profile
                                break 
                            elif selected_depth_profile is None: # Take any format if Y16 not found yet for target
                                selected_depth_profile = video_profile


                        # Keep track of the highest resolution profile found so far
                        if current_resolution > highest_resolution:
                            highest_resolution = current_resolution
                            best_profile = video_profile
                        elif current_resolution == highest_resolution:
                            # If resolutions are the same, prefer Y16 format
                            if video_profile.get_format() == OBFormat.Y16 and (best_profile is None or best_profile.get_format() != OBFormat.Y16):
                                best_profile = video_profile


            if selected_depth_profile:
                self.depth_profile = selected_depth_profile
                print(f"Selected Target Depth Profile: {self.depth_profile.get_width()}x{self.depth_profile.get_height()}@{self.depth_profile.get_fps()}fps {self.depth_profile.get_format()}")
            elif best_profile:
                self.depth_profile = best_profile
                print(f"Selected Highest Resolution Depth Profile: {self.depth_profile.get_width()}x{self.depth_profile.get_height()}@{self.depth_profile.get_fps()}fps {self.depth_profile.get_format()}")
            else:
                # Fallback to default if no profiles found or list is empty (should not happen if previous check passes)
                self.depth_profile = profile_list_depth.get_default_video_stream_profile()
                if self.depth_profile is None:
                    raise RuntimeError("Failed to get any depth video stream profile.")
                print(f"Fallback to Default Depth Profile: {self.depth_profile.get_width()}x{self.depth_profile.get_height()}@{self.depth_profile.get_fps()}fps {self.depth_profile.get_format()}")
            
            if self.depth_profile is None: # Should be caught by RuntimeError above, but as a safeguard
                raise RuntimeError("Failed to select a depth video stream profile.")
            # print( # Original print for default, now adapted
            #    f"Selected Depth Profile: {self.depth_profile.get_width()}x{self.depth_profile.get_height()}@{self.depth_profile.get_fps()}fps {self.depth_profile.get_format()}")
        except OBError as e:
            raise RuntimeError(f"Error getting depth profiles: {e}")

        try:
            profile_list_color = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list_color is None:
                print("Warning: Failed to get color stream profile list (Color sensor may be missing or disabled).")
                self.color_profile = None
            else:
                self.color_profile = profile_list_color.get_default_video_stream_profile()
                if self.color_profile is None:
                    print("Warning: Failed to get default color video stream profile.")
                else:
                    print(
                        f"Default Color Profile: {self.color_profile.get_width()}x{self.color_profile.get_height()}@{self.color_profile.get_fps()}fps {self.color_profile.get_format()}")
        except OBError as e:
            print(f"Warning: Error getting color profiles: {e}")
            self.color_profile = None
        # --- End Profile Getting ---

        self.extrinsic_matrix = []

        # --- Defer starting stream and getting params until start_stream() is called ---
        # self.start_stream() # Don't auto-start here

        # Initialize intrinsic/extrinsic placeholders
        self.depth_fx = self.depth_fy = self.depth_ppx = self.depth_ppy = None
        self.depth_distortion = None
        self.rgb_fx = self.rgb_fy = self.rgb_ppx = self.rgb_ppy = None
        self.rgb_distortion = None
        self.stream = False

    def color_viewer(self):
        if not self.stream:
            print("Stream not started. Call start_stream() first.")
            # Attempt to start with default settings if needed
            try:
                self.start_stream()
            except Exception as e:
                print(f"Failed to auto-start stream for color_viewer: {e}")
                return

        while True:
            try:
                frames: FrameSet = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                # covert to RGB format
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("failed to convert frame to image")
                    continue
                cv2.imshow("Color Viewer", color_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in color_viewer loop: {e}")
                break  # Exit loop on error
        # Don't stop the pipeline here, let the caller manage it
        # self.stop()
        cv2.destroyWindow("Color Viewer")  # Close only this window

    def get_device_name(self):
        return self.device_info.get_name()

    def get_device_pid(self):
        return self.device_info.get_pid()

    def get_serial_number(self):
        return self.device_info.get_serial_number()

    def set_auto_exposure(self, auto_exposure: bool):
        try:
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, auto_exposure)
            print(f"Auto exposure set to: {auto_exposure}")
        except OBError as e:
            print(f"Failed to set auto exposure: {e}")

    def get_current_exposure(self):
        try:
            return self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
        except OBError as e:
            print(f"Failed to get current exposure: {e}")
            return None

    def set_exposure(self, exposure_value: int):
        try:
            # Optionally disable auto exposure first
            self.set_auto_exposure(False)
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_value)
            print(f"Exposure set to: {exposure_value}")
        except OBError as e:
            print(f"Failed to set exposure: {e}")

    def adjust_exposure(self, adjustment: int):
        curr_exposure = self.get_current_exposure()
        if curr_exposure is not None:
            new_exposure = curr_exposure + adjustment
            # Add bounds checking if necessary based on device limits
            self.set_exposure(new_exposure)

    def set_software_filter(self, soft_filter: bool):
        try:
            self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, soft_filter)
            print(f"Software filter set to: {soft_filter}")
        except OBError as e:
            print(f"Failed to set software filter: {e}")

    def reboot(self):
        try:
            print("Rebooting device...")
            self.device.reboot()
            print("Device reboot command sent. Re-initialization might be needed.")
            self.stream = False  # Mark stream as stopped
        except OBError as e:
            print(f"Failed to reboot device: {e}")

    def start_stream(self, depth_stream=True, color_stream=True, enable_sync=True, use_alignment=True):
        '''Configures and starts the pipeline streams.'''
        self.config = Config()  # Reset config
        if color_stream and self.color_profile:
            print("Enabling color stream...")
            self.config.enable_stream(self.color_profile)
        elif color_stream and not self.color_profile:
            print("Warning: Color stream requested but no valid profile found.")

        if depth_stream and self.depth_profile:
            print("Enabling depth stream...")
            self.config.enable_stream(self.depth_profile)
            if use_alignment and color_stream and self.color_profile:
                print("Setting alignment mode to SW_MODE (Depth to Color).")
                self.config.set_align_mode(OBAlignMode.SW_MODE)  # Align Depth to Color
            elif use_alignment:
                print("Warning: Alignment requested but color stream is disabled or unavailable. Alignment not set.")
        elif depth_stream and not self.depth_profile:
            print("Error: Depth stream requested but no valid profile found.")
            # Maybe raise an error here if depth is essential?

        if not (depth_stream and self.depth_profile) and not (color_stream and self.color_profile):
            raise RuntimeError("No valid streams selected or available to start.")

        print("Starting pipeline...")
        try:
            self.pipeline.start(self.config)
        except OBError as e:
            raise RuntimeError(f"Failed to start pipeline: {e}")

        self.stream = True
        print("Pipeline started.")

        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
                print("Frame sync enabled.")
            except OBError as e:
                print(f"Warning: Failed to enable frame sync: {e}")
            except Exception as e:
                print(f"Warning: An unexpected error occurred enabling frame sync: {e}")

        # --- Get camera parameters AFTER starting pipeline ---
        try:
            self.param = self.pipeline.get_camera_param()
            print("Camera parameters obtained.")
            # print(self.param) # Optionally print details

            # Update intrinsic/extrinsic variables
            self.depth_fx = self.param.depth_intrinsic.fx
            self.depth_fy = self.param.depth_intrinsic.fy
            self.depth_ppx = self.param.depth_intrinsic.cx
            self.depth_ppy = self.param.depth_intrinsic.cy
            self.depth_distortion = self.param.depth_distortion

            self.rgb_fx = self.param.rgb_intrinsic.fx
            self.rgb_fy = self.param.rgb_intrinsic.fy
            self.rgb_ppx = self.param.rgb_intrinsic.cx
            self.rgb_ppy = self.param.rgb_intrinsic.cy
            self.rgb_distortion = self.param.rgb_distortion
            # print(self.param) # Can be verbose, print if needed

        except OBError as e:
            print(f"Warning: Failed to get camera parameters after starting pipeline: {e}")
            self.param = None  # Ensure param is None if fetching failed

        # --- Initialize PointCloudFilter AFTER getting parameters ---
        if self.param is not None:
            try:
                self.point_cloud_filter = PointCloudFilter()
                self.point_cloud_filter.set_camera_param(self.param)
                print("PointCloudFilter created and configured.")
            except OBError as e:
                print(f"Error creating PointCloudFilter: {e}")
                self.point_cloud_filter = None
        else:
            print("Warning: Cannot initialize PointCloudFilter because camera parameters are missing.")
            self.point_cloud_filter = None

    def get_frames(self):
        '''Gets color image, depth data, and raw depth frame after alignment.'''
        if not self.stream:
            print("Stream not started. Call start_stream() first.")
            return None, None, None

        frames = None
        try:
            frames = self.pipeline.wait_for_frames(100)  # Timeout in milliseconds
        except OBError as e:
            print(f"Error waiting for frames: {e}")
            return None, None, None

        if frames is None:
            # print("No frames received (timeout).") # Can be noisy
            return None, None, None

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Basic check if frames are missing
        if depth_frame is None:
            # print("Depth frame missing in frameset.") # Can be noisy
            return None, None, None
        # Color frame might be optional depending on config
        # if color_frame is None:
        #      print("Color frame missing in frameset.")

        # --- Get point cloud (This part seems redundant if the goal is images/depth) ---
        # point_clouds = frames.get_point_cloud(self.param) # Deprecated/less common usage
        # if point_clouds is None:
        #     print("No point clouds")
        # --- End redundant part ---

        color_image = None
        if color_frame is not None:
            color_image = self.color_frame2color_image(color_frame)

        depth_data = self.depth_frame2depth_data(depth_frame, filter_on=True)

        return color_image, depth_data, depth_frame  # Return processed depth, not raw frame data maybe?

    def is_streaming(self):
        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ NEW FUNCTION TO GET POINT CLOUD +++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_point_cloud(self, colored=True):
        """
        Waits for frames, computes point cloud, filters by depth range, and returns it.

        Args:
            colored (bool): If True, attempts to generate an RGB point cloud (XYZRGB).
                            Requires color stream enabled and alignment.
                            If False or color/alignment unavailable, generates XYZ.

        Returns:
            np.ndarray | None: An Nx3 (XYZ) or Nx6 (XYZRGB) NumPy array filtered by depth,
                               or None if generation/filtering fails or yields no points.
        """
        if not self.stream:
            print("get_point_cloud Error: Stream not started. Call start_stream() first.")
            return None
        if self.point_cloud_filter is None:
            print("get_point_cloud Error: PointCloudFilter is not initialized (check start_stream).")
            return None

        frames = None
        try:
            # 1. Wait for frames
            frames = self.pipeline.wait_for_frames(100)
            if frames is None: return None

            # 2. Get depth frame
            depth_frame = frames.get_depth_frame()
            if depth_frame is None: return None

            # 3. Get color frame & determine format
            color_frame = frames.get_color_frame() if colored else None
            has_valid_color_for_pc = colored and (color_frame is not None)
            # (Alignment check logic remains the same)
            alignment_enabled = self.config.get_align_mode() != OBAlignMode.ALIGN_DISABLE if hasattr(self.config,
                                                                                                     'get_align_mode') else True
            if colored and not has_valid_color_for_pc:
                print("get_point_cloud: Warning - Color requested, but color frame missing. Falling back to XYZ.")
            elif colored and not alignment_enabled:
                print("get_point_cloud: Warning - Color requested, but alignment seems disabled. Falling back to XYZ.")
                has_valid_color_for_pc = False

            # 4. Configure PointCloudFilter
            scale = depth_frame.get_depth_scale()
            self.point_cloud_filter.set_position_data_scaled(scale)
            point_format = OBFormat.RGB_POINT if has_valid_color_for_pc else OBFormat.POINT
            self.point_cloud_filter.set_create_point_format(point_format)

            # 5. Process frames
            point_cloud_frame = self.point_cloud_filter.process(frames)
            if point_cloud_frame is None: return None

            # 6. Calculate points
            points_data = self.point_cloud_filter.calculate(point_cloud_frame)
            if points_data is None: return None
            points = np.array(points_data)

            # 7. Validate initial shape
            if points.size == 0: return None
            expected_cols = 6 if point_format == OBFormat.RGB_POINT else 3
            if points.ndim != 2 or points.shape[1] != expected_cols:
                print(
                    f"get_point_cloud: Error - Unexpected points shape: {points.shape}, expected {expected_cols} cols.")
                # (Salvaging logic remains the same)
                if points.shape[1] == 3 and point_format == OBFormat.RGB_POINT:
                    return points
                elif points.shape[1] == 6 and point_format == OBFormat.POINT:
                    return points[:, :3]
                else:
                    return None

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++ ADD DEPTH FILTERING BASED ON Z COORDINATE ++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Extract Z coordinates (typically the 3rd column, index 2)
            z_coordinates = points[:, 2]

            # Create a boolean mask for points within the desired depth range (in meters)
            # Using class constants MIN_DEPTH_M and MAX_DEPTH_M defined above
            depth_mask = (z_coordinates >= MIN_DEPTH) & (z_coordinates <= MAX_DEPTH)

            # Apply the mask to filter the points
            filtered_points = points[depth_mask]

            # Check if any points remain after filtering
            if filtered_points.size == 0:
                # print("get_point_cloud: No points remaining after depth filtering.") # Can be noisy
                return None  # Return None if the filter removed all points
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++ END DEPTH FILTERING ++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # 8. Return the FILTERED result
            return filtered_points  # Return the depth-filtered points

        except OBError as e:
            print(f"get_point_cloud: Orbbec SDK Error - {e}")
            return None
        except Exception as e:
            print(f"get_point_cloud: Unexpected Error - {e}")
            import traceback
            traceback.print_exc()  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # +++ END NEW FUNCTION ++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_depth_for_color_pixel(self, depth_frame, color_point, show=False):
        # Ensure depth_frame is valid
        if depth_frame is None:
            print("get_depth_for_color_pixel Error: Invalid depth_frame provided.")
            return 0

        # Use the consistent method to get depth data
        depth_data = self.depth_frame2depth_data(depth_frame, filter_on=True)  # Apply filtering/clipping

        if show:
            import matplotlib.pyplot as plt
            # Use a copy for visualization to avoid modifying original depth_data if needed later
            depth_display = depth_data.copy()
            # Normalize non-zero values for better visualization if range is large
            mask = depth_display > 0
            if np.any(mask):
                min_val = depth_display[mask].min()
                max_val = depth_display[mask].max()
                if max_val > min_val:
                    # Normalize only the valid range to 0-255
                    depth_display[mask] = ((depth_display[mask] - min_val) / (max_val - min_val)) * 255
                else:
                    depth_display[mask] = 128  # Assign mid-gray if all valid depths are the same
            depth_display = depth_display.astype(np.uint8)
            depth_image_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            # Set zero depth to black after colormap
            depth_image_color[~mask] = [0, 0, 0]

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(depth_image_color, cv2.COLOR_BGR2RGB))
            plt.title("Depth Viewer (Filtered & Clipped, Zeros=Black)")
            plt.axis("off")
            # Add marker for the target pixel
            plt.scatter([color_point[0]], [color_point[1]], c='red', s=40, marker='x')  # Draw marker
            plt.show()

        center_x, center_y = int(round(color_point[0])), int(round(color_point[1]))

        # Check bounds
        if not (0 <= center_y < depth_data.shape[0] and 0 <= center_x < depth_data.shape[1]):
            print(
                f"get_depth_for_color_pixel Error: Pixel ({center_x}, {center_y}) is out of depth bounds ({depth_data.shape[1]}, {depth_data.shape[0]})")
            return 0

        center_depth = depth_data[center_y, center_x]

        # If center depth is 0 (invalid/clipped), search neighbors
        if center_depth == 0:
            search_radius = 1
            max_radius = 10  # Limit search radius to avoid excessive computation
            found_depth = 0
            while search_radius <= max_radius:
                min_x = max(center_x - search_radius, 0)
                max_x = min(center_x + search_radius + 1, depth_data.shape[1])
                min_y = max(center_y - search_radius, 0)
                max_y = min(center_y + search_radius + 1, depth_data.shape[0])

                # Extract neighborhood and find non-zero minimum or average? Let's find first non-zero.
                neighborhood = depth_data[min_y:max_y, min_x:max_x]
                valid_depths = neighborhood[neighborhood > 0]

                if valid_depths.size > 0:
                    # Option 1: Take the first one found (can be arbitrary)
                    # found_depth = valid_depths[0]

                    # Option 2: Take the minimum valid depth in neighborhood
                    found_depth = valid_depths.min()

                    # Option 3: Take the average valid depth (might smooth too much)
                    # found_depth = valid_depths.mean()

                    break  # Exit while loop once a valid depth is found

                search_radius += 1

            if found_depth > 0:
                # print(f"Original pixel depth was 0, found nearby depth: {found_depth} at radius {search_radius}")
                center_depth = found_depth
            else:
                # print(f"No valid depth found within radius {max_radius} of pixel ({center_x}, {center_y}).")
                center_depth = 0  # Return 0 if nothing found

        return center_depth  # Return depth in mm (as uint16)

    def color_frame2color_image(self, color_frame):
        if color_frame is None: return None
        try:
            color_image = frame_to_bgr_image(color_frame)
            return color_image
        except Exception as e:
            print(f"Error converting color frame to image: {e}")
            return None

    def depth_frame2depth_data(self, depth_frame, filter_on=True):
        if depth_frame is None: return None
        try:
            height = depth_frame.get_height()
            width = depth_frame.get_width()
            scale = depth_frame.get_depth_scale()  # Typically meters/unit (e.g., 0.001)

            # Get data as uint16
            depth_data_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data_u16 = depth_data_u16.reshape((height, width))

            # Convert to float32 and apply scale (Result is usually in meters)
            depth_data_m = depth_data_u16.astype(np.float32) * scale

            # Apply min/max depth filtering (convert MM thresholds to Meters)
            min_depth_m = MIN_DEPTH / 1000.0
            max_depth_m = MAX_DEPTH / 1000.0
            depth_data_m = np.where((depth_data_m > min_depth_m) & (depth_data_m < max_depth_m), depth_data_m, 0)

            # Convert back to uint16 millimeters for consistency with original intent? Or keep as meters?
            # Let's return uint16 millimeters as the function `get_depth_for_color_pixel` seems to expect it.
            depth_data_final_mm = (depth_data_m * 1000).astype(np.uint16)

            if filter_on:
                # Apply temporal filtering (expects uint16 mm based on previous context)
                depth_data_final_mm = self.temporal_filter.process(depth_data_final_mm)

            return depth_data_final_mm
        except Exception as e:
            print(f"Error processing depth frame: {e}")
            return None

    def show_depth_frame(self, depth_frame, window_name="Depth Viewer"):
        # Get depth data in uint16 mm format, apply temporal filter
        depth_data_mm = self.depth_frame2depth_data(depth_frame, filter_on=True)
        if depth_data_mm is None:
            print("Cannot show depth frame, processing failed.")
            return

        # Normalize for visualization (use the mm data)
        # Handle potential empty frame after filtering
        if depth_data_mm.size == 0: return

        mask = depth_data_mm > 0
        depth_display = np.zeros_like(depth_data_mm, dtype=np.uint8)  # Start with black background

        if np.any(mask):
            min_val = depth_data_mm[mask].min()
            max_val = depth_data_mm[mask].max()
            if max_val > min_val:
                # Normalize only the valid range (min_val to max_val in mm) to 0-255
                norm_data = ((depth_data_mm[mask] - min_val) / (max_val - min_val)) * 255
                depth_display[mask] = norm_data.astype(np.uint8)
            elif min_val > 0:  # All valid depths are the same non-zero value
                depth_display[mask] = 128  # Assign mid-gray

        depth_image_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        # Ensure zeros remain black after colormap
        depth_image_color[~mask] = [0, 0, 0]

        cv2.imshow(window_name, depth_image_color)

    def stop(self):
        if self.stream:
            print("Stopping pipeline...")
            try:
                self.pipeline.stop()
                self.stream = False
                print("Pipeline stopped.")
            except OBError as e:
                print(f"Error stopping pipeline: {e}")
        else:
            print("Pipeline already stopped.")

    def adjust_exposure_based_on_brightness(self, target_brightness=100, max_iterations=10):
        # Requires stream to be running
        if not self.stream:
            print("adjust_exposure Error: Stream not started.")
            return

        # Set adjustment step and exposure bounds (consider querying device limits if possible)
        exposure_step = 10
        min_exposure = 10  # Example min
        max_exposure = 10000  # Example max
        tolerance = 10  # Brightness tolerance

        print(f"Adjusting exposure towards target brightness: {target_brightness} +/- {tolerance}")

        for i in range(max_iterations):
            # Get color frame only to check brightness
            frames = self.pipeline.wait_for_frames(100)
            if frames is None: continue
            color_frame = frames.get_color_frame()
            if color_frame is None: continue
            color_image = self.color_frame2color_image(color_frame)
            if color_image is None: continue

            # Calculate brightness (using weighted average)
            # Ensure image is not empty
            if color_image.size == 0: continue
            current_brightness = (
                    0.299 * color_image[:, :, 2].mean() +  # Red
                    0.587 * color_image[:, :, 1].mean() +  # Green
                    0.114 * color_image[:, :, 0].mean()  # Blue
            )

            print(f"Iteration {i + 1}/{max_iterations}: Current brightness = {current_brightness:.2f}")

            current_exposure = self.get_current_exposure()
            if current_exposure is None:
                print("Could not get current exposure, aborting adjustment.")
                break

            # Check if within tolerance
            if abs(current_brightness - target_brightness) <= tolerance:
                print("Brightness within target range. Adjustment complete.")
                break

            # Adjust exposure
            if current_brightness < target_brightness:
                new_exposure = min(current_exposure + exposure_step, max_exposure)
                print(f"  Brightness low. Increasing exposure from {current_exposure} to {new_exposure}")
            else:  # current_brightness > target_brightness
                new_exposure = max(current_exposure - exposure_step, min_exposure)
                print(f"  Brightness high. Decreasing exposure from {current_exposure} to {new_exposure}")

            # Apply if changed
            if new_exposure != current_exposure:
                self.set_exposure(new_exposure)
                time.sleep(0.1)  # Wait a bit for exposure change to take effect
            else:
                print("  Exposure already at limit, cannot adjust further in this direction.")
                break  # Stop if hitting limits

        else:  # Loop finished without break (max_iterations reached)
            print(f"Adjustment stopped after {max_iterations} iterations.")

    def load_extrinsic(self):
        """从配置文件中加载外参."""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                if config_data and 'hand_eye_transformation_matrix' in config_data:
                    transformation_matrix = config_data['hand_eye_transformation_matrix']
                    self.extrinsic_matrix = np.array(transformation_matrix)
                    print("Extrinsic matrix loaded successfully.")
                    # print(self.extrinsic_matrix)
                else:
                    print(f"Error: 'hand_eye_transformation_matrix' not found in {self.config_path}")
                    self.extrinsic_matrix = np.identity(4)  # Default to identity? Or None?

        except FileNotFoundError:
            print(f"Error: Extrinsic config file not found at {self.config_path}")
            self.extrinsic_matrix = np.identity(4)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {self.config_path}: {e}")
            self.extrinsic_matrix = np.identity(4)
        except Exception as e:
            print(f"加载外参时发生未知错误: {e}")
            self.extrinsic_matrix = np.identity(4)


# === Utility functions remain outside the class ===

def get_serial_numbers():
    """Gets serial numbers of all connected Orbbec devices."""
    serial_numbers = []
    try:
        context = Context()
        device_list = context.query_devices()
        count = device_list.get_count()
        print(f"Found {count} Orbbec device(s).")
        for i in range(count):
            try:
                sn = device_list.get_device_serial_number_by_index(i)
                if sn:
                    serial_numbers.append(sn)
                else:
                    print(f"Warning: Could not get serial number for device at index {i}.")
            except OBError as e:
                print(f"Error getting serial number for device at index {i}: {e}")
    except OBError as e:
        print(f"Error querying devices: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in get_serial_numbers: {e}")
    return serial_numbers


def initialize_connected_cameras(serial_number):
    """Initializes a single camera by serial number."""
    try:
        device = OrbbecCamera(serial_number)
        return device  # Return the single Camera object
    except Exception as e:
        print(f"Failed to initialize camera with SN {serial_number}: {e}")
        return None


def initialize_all_connected_cameras(serial_numbers=None):
    """Initializes cameras for the given serial numbers, or all found if None."""
    if serial_numbers is None:
        print("No serial numbers provided, attempting to find all connected devices...")
        serial_numbers = get_serial_numbers()
        if not serial_numbers:
            print("No devices found.")
            return []

    if isinstance(serial_numbers, str):
        serial_numbers = [serial_numbers]

    devices = []
    print(f"Attempting to initialize cameras for SNs: {serial_numbers}")
    for sn in serial_numbers:
        print(f"  Initializing camera {sn}...")
        try:
            camera = OrbbecCamera(sn)
            devices.append(camera)
            print(f"  Successfully initialized camera {sn}.")
        except Exception as e:
            print(f"  FAILED to initialize camera {sn}: {e}")
            # Optionally continue to initialize others, or stop? Let's continue.
    return devices


def close_connected_cameras(cameras):
    """Stops the pipeline for all provided camera objects."""
    if isinstance(cameras, OrbbecCamera):  # Handle single camera object
        cameras = [cameras]
    print(f"Closing {len(cameras)} camera(s)...")
    for camera in cameras:
        try:
            camera.stop()
        except Exception as e:
            print(f"Error stopping camera {camera.get_serial_number()}: {e}")


# === Main execution block ===
def main():
    # Get available serial numbers
    available_sns = get_serial_numbers()

    # Initialize cameras (use all found, or specify a list)
    cameras = initialize_all_connected_cameras(available_sns[0])

    if not cameras:
        print("No cameras were initialized successfully. Exiting.")
        return

    # Start streams for all initialized cameras
    print("\nStarting streams for all cameras...")
    for camera in cameras:
        try:
            # Start with depth, color, alignment, and sync enabled
            camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        except Exception as e:
            print(f"Failed to start stream for camera {camera.get_serial_number()}: {e}")
            # Optionally remove this camera from the list if stream fails?
            # cameras.remove(camera) # Be careful modifying list while iterating

    if not cameras:  # Check again if any streams failed
        print("No camera streams started successfully. Exiting.")
        return

    print("\nEntering main loop (Press 'q' in OpenCV window to quit)...")
    try:
        while True:
            quit_flag = False
            for idx, camera in enumerate(cameras):
                if not camera.stream:  # Skip cameras where stream didn't start
                    continue

                # --- Option 1: Get Images/Depth Data ---
                # color_image, depth_data, _ = camera.get_frames()
                # if color_image is not None:
                #     cv2.imshow(f"Cam {idx} ({camera.get_serial_number()}) Color", color_image)
                # if depth_data is not None:
                #      # Need to visualize depth_data (uint16 mm) correctly
                #      pass # Add visualization if needed

                # --- Option 2: Get Point Cloud ---
                points = camera.get_point_cloud(colored=True)  # Get XYZRGB
                # points = camera.get_point_cloud(colored=False) # Get XYZ only

                if points is not None:
                    print(f"Cam {idx} ({camera.get_serial_number()}) Point Cloud Shape: {points.shape}")
                    # --- Visualization of Point Cloud (Requires Open3D) ---
                    try:
                        import open3d as o3d
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use first 3 columns for XYZ
                        if points.shape[1] == 6:  # If we have color data
                            # Assuming colors are in 0-255 range from SDK (check this assumption)
                            colors = points[:, 3:6] / 255.0
                            colors = np.clip(colors, 0.0, 1.0)  # Ensure valid range
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        else:
                            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Default gray

                        if not pcd.has_points():
                            print(f"  Cam {idx}: Point cloud is empty after processing. Skipping visualization.")
                            continue  # Skip to next camera if no points

                        # --- MODIFIED VISUALIZATION WITH POINT SIZE CONTROL ---
                        print(f"  Visualizing point cloud for Cam {idx} (Close window to continue)...")

                        vis = o3d.visualization.Visualizer()
                        vis.create_window(window_name=f"Cam {idx} Point Cloud", width=600, height=400)

                        render_option = vis.get_render_option()
                        render_option.point_size = 1.0  # <<< --- 设置点的大小在这里 ---
                        # render_option.background_color = np.asarray([0, 0, 0]) # 可选：设置背景颜色为黑色

                        vis.add_geometry(pcd)
                        vis.run()  # 阻塞式运行，直到窗口关闭
                        vis.destroy_window()

                        print(f"  Visualization window for Cam {idx} closed.")
                        # --- END MODIFIED VISUALIZATION ---

                    except ImportError:
                        if idx == 0: print("Open3D not installed. Skipping point cloud visualization.")
                    except Exception as e_vis:
                        print(f"Error during Open3D visualization for Cam {idx}: {e_vis}")
                    # --- End Visualization ---
                else:
                    # print(f"Cam {idx} ({camera.get_serial_number()}): Failed to get point cloud this cycle.") # Can be noisy
                    pass

            # Check for quit key (assuming at least one OpenCV window is potentially open)
            # Note: If Open3D window is active and blocking, cv2.waitKey might not register immediately.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                print("Quit key pressed.")
                quit_flag = True
                # break  # This breaks inner loop (camera loop)

            if quit_flag:  # Check quit_flag after iterating through all cameras for this cycle
                break  # Exit outer loop (while True)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting loop.")
    finally:
        # Cleanup
        print("\nCleaning up...")
        close_connected_cameras(cameras)  # Stop pipelines
        cv2.destroyAllWindows()  # Close any OpenCV windows
        print("Program finished.")


if __name__ == "__main__":
    main()
