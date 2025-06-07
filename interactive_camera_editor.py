#!/usr/bin/env python3
"""
交互式相机编辑器
允许用户自定义相机位置、调整拍摄角度，并实时预览效果
"""

import open3d as o3d
import numpy as np
import copy
import json
import os
import time
from typing import List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import warnings

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 配置中文字体 - 使用系统实际可用的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


class CameraConfig:
    """相机配置类"""
    
    def __init__(self):
        self.cameras = []  # [(position, look_at, fov, name), ...]
        self.object_center = np.array([0, 0, 0])
        self.object_size = 1.0
        
    def add_camera(self, position: np.ndarray, look_at: np.ndarray, fov: float = 50.0, name: str = "Camera"):
        """添加相机"""
        self.cameras.append({
            'position': position.copy(),
            'look_at': look_at.copy(),
            'fov': fov,
            'name': name
        })
    
    def remove_camera(self, index: int):
        """删除相机"""
        if 0 <= index < len(self.cameras):
            del self.cameras[index]
    
    def update_camera(self, index: int, position: np.ndarray = None, look_at: np.ndarray = None, fov: float = None, name: str = None):
        """更新相机参数"""
        if 0 <= index < len(self.cameras):
            if position is not None:
                self.cameras[index]['position'] = position.copy()
            if look_at is not None:
                self.cameras[index]['look_at'] = look_at.copy()
            if fov is not None:
                self.cameras[index]['fov'] = fov
            if name is not None:
                self.cameras[index]['name'] = name
    
    def save_config(self, filename: str):
        """保存配置到文件"""
        config = {
            'object_center': self.object_center.tolist(),
            'object_size': self.object_size,
            'cameras': []
        }
        
        for cam in self.cameras:
            config['cameras'].append({
                'position': cam['position'].tolist(),
                'look_at': cam['look_at'].tolist(),
                'fov': cam['fov'],
                'name': cam['name']
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filename: str):
        """从文件加载配置"""
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.object_center = np.array(config['object_center'])
        self.object_size = config['object_size']
        self.cameras = []
        
        for cam_data in config['cameras']:
            self.cameras.append({
                'position': np.array(cam_data['position']),
                'look_at': np.array(cam_data['look_at']),
                'fov': cam_data['fov'],
                'name': cam_data['name']
            })


class InteractiveCameraEditor:
    """交互式相机编辑器"""
    
    def __init__(self, stl_file: str = None):
        self.stl_file = stl_file
        self.mesh = None
        self.config = CameraConfig()
        self.selected_camera_index = -1
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("交互式相机编辑器")
        self.root.geometry("1200x800")
        
        # 加载STL文件
        if stl_file and os.path.exists(stl_file):
            self.load_stl_file(stl_file)
        
        self.setup_ui()
        self.update_camera_list()
        self.update_3d_plot()
    
    def load_stl_file(self, filename: str):
        """加载STL文件"""
        try:
            self.mesh = o3d.io.read_triangle_mesh(filename)
            if self.mesh.has_vertices():
                bbox = self.mesh.get_axis_aligned_bounding_box()
                self.config.object_center = bbox.get_center()
                self.config.object_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)
                
                # 添加默认相机设置
                self.add_default_cameras()
                print(f"成功加载STL文件: {filename}")
            else:
                print(f"错误：无法读取STL文件 {filename}")
        except Exception as e:
            print(f"加载STL文件时出错: {e}")
    
    def add_default_cameras(self):
        """添加默认相机设置 - 前后左右4个相机"""
        center = self.config.object_center
        radius = self.config.object_size * 1.8 if self.config.object_size > 0 else 120
        elevation_height = self.config.object_size * 0.2 if self.config.object_size > 0 else 20
        
        # 定义四个方向：前、后、左、右
        directions = [
            {"name": "前方", "angle": 0},      # 前方 (0度)
            {"name": "右方", "angle": 90},     # 右方 (90度) 
            {"name": "后方", "angle": 180},    # 后方 (180度)
            {"name": "左方", "angle": 270}     # 左方 (270度)
        ]
        
        for direction in directions:
            angle_rad = np.radians(direction["angle"])
            
            # 计算相机位置 (稍微仰视)
            x = center[0] + radius * np.cos(angle_rad)
            y = center[1] + radius * np.sin(angle_rad)
            z = center[2] + elevation_height
            
            position = np.array([x, y, z])
            
            # 计算朝向物体中心的方向
            look_at = center - position
            if np.linalg.norm(look_at) > 0:
                look_at = look_at / np.linalg.norm(look_at)
            else:
                look_at = np.array([0, 0, -1])
            
            name = f"{direction['name']}相机"
            self.config.add_camera(position, look_at, 50.0, name)
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # 右侧3D视图
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(left_frame)
        self.setup_3d_view(right_frame)
    
    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 文件操作
        file_frame = ttk.LabelFrame(parent, text="文件操作")
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="加载STL文件", command=self.load_stl_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="保存相机配置", command=self.save_config_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="加载相机配置", command=self.load_config_dialog).pack(fill=tk.X, pady=2)
        
        # 相机列表
        camera_frame = ttk.LabelFrame(parent, text="相机列表")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 相机列表框
        list_frame = ttk.Frame(camera_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.camera_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        self.camera_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.camera_listbox.bind('<<ListboxSelect>>', self.on_camera_select)
        
        scrollbar.config(command=self.camera_listbox.yview)
        
        # 相机操作按钮
        button_frame = ttk.Frame(camera_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="添加", command=self.add_camera_dialog, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="删除", command=self.delete_selected_camera, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="复制", command=self.copy_selected_camera, width=8).pack(side=tk.LEFT, padx=2)
        
        # 相机参数编辑
        edit_frame = ttk.LabelFrame(parent, text="相机参数")
        edit_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 相机名称
        ttk.Label(edit_frame, text="名称:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(edit_frame, textvariable=self.name_var, width=20)
        self.name_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # 位置
        ttk.Label(edit_frame, text="位置 X:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.pos_x_var = tk.DoubleVar()
        self.pos_x_entry = ttk.Entry(edit_frame, textvariable=self.pos_x_var, width=10)
        self.pos_x_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(edit_frame, text="Y:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pos_y_var = tk.DoubleVar()
        self.pos_y_entry = ttk.Entry(edit_frame, textvariable=self.pos_y_var, width=10)
        self.pos_y_entry.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(edit_frame, text="Z:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.pos_z_var = tk.DoubleVar()
        self.pos_z_entry = ttk.Entry(edit_frame, textvariable=self.pos_z_var, width=10)
        self.pos_z_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # 视场角
        ttk.Label(edit_frame, text="视场角:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.fov_var = tk.DoubleVar(value=50.0)
        self.fov_entry = ttk.Entry(edit_frame, textvariable=self.fov_var, width=10)
        self.fov_entry.grid(row=4, column=1, padx=5, pady=2)
        
        # 更新按钮
        ttk.Button(edit_frame, text="更新相机", command=self.update_selected_camera).grid(row=5, column=0, columnspan=2, pady=10)
        
        # 快速设置
        quick_frame = ttk.LabelFrame(parent, text="快速设置")
        quick_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(quick_frame, text="前后左右4相机", command=self.add_four_direction_cameras).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="环绕相机(8个)", command=lambda: self.add_circular_cameras(8)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="环绕相机(12个)", command=lambda: self.add_circular_cameras(12)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="清除所有相机", command=self.clear_all_cameras).pack(fill=tk.X, pady=2)
        
        # 生成点云
        generate_frame = ttk.LabelFrame(parent, text="生成点云")
        generate_frame.pack(fill=tk.X)
        
        ttk.Button(generate_frame, text="生成多角度点云", command=self.generate_pointcloud).pack(fill=tk.X, pady=2)
    
    def setup_3d_view(self, parent):
        """设置3D视图"""
        # 创建matplotlib图形
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
    
    def update_3d_plot(self):
        """更新3D图形"""
        self.ax.clear()
        
        # 绘制物体中心
        center = self.config.object_center
        self.ax.scatter(center[0], center[1], center[2], color='red', s=100, label='物体中心')
        
        # 绘制相机位置
        if self.config.cameras:
            camera_positions = np.array([cam['position'] for cam in self.config.cameras])
            self.ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                          color='blue', s=50, label='相机位置')
            
            # 绘制选中的相机
            if 0 <= self.selected_camera_index < len(self.config.cameras):
                selected_cam = self.config.cameras[self.selected_camera_index]
                pos = selected_cam['position']
                self.ax.scatter(pos[0], pos[1], pos[2], color='yellow', s=150, marker='*', label='选中相机')
                
                # 绘制视线方向
                look_direction = selected_cam['look_at']
                end_pos = pos + look_direction * self.config.object_size * 0.3
                self.ax.plot([pos[0], end_pos[0]], [pos[1], end_pos[1]], [pos[2], end_pos[2]], 
                           'y-', linewidth=3, label='视线方向')
            
            # 绘制所有相机的视线方向（淡一些）
            for i, cam in enumerate(self.config.cameras):
                if i != self.selected_camera_index:
                    pos = cam['position']
                    look_direction = cam['look_at']
                    end_pos = pos + look_direction * self.config.object_size * 0.2
                    self.ax.plot([pos[0], end_pos[0]], [pos[1], end_pos[1]], [pos[2], end_pos[2]], 
                               'g--', alpha=0.3, linewidth=1)
        
        # 设置坐标轴
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title(f'相机设置 ({len(self.config.cameras)}个相机)')
        
        # 设置相等的坐标轴比例
        if self.config.object_size > 0:
            max_range = self.config.object_size * 1.5
            center = self.config.object_center
            self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
            self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
            self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        self.canvas.draw()
    
    def update_camera_list(self):
        """更新相机列表"""
        self.camera_listbox.delete(0, tk.END)
        for i, cam in enumerate(self.config.cameras):
            self.camera_listbox.insert(tk.END, f"{i+1}. {cam['name']}")
    
    def on_camera_select(self, event):
        """相机选择事件"""
        selection = self.camera_listbox.curselection()
        if selection:
            self.selected_camera_index = selection[0]
            self.load_camera_params()
            self.update_3d_plot()
    
    def load_camera_params(self):
        """加载选中相机的参数到编辑框"""
        if 0 <= self.selected_camera_index < len(self.config.cameras):
            cam = self.config.cameras[self.selected_camera_index]
            self.name_var.set(cam['name'])
            self.pos_x_var.set(round(cam['position'][0], 2))
            self.pos_y_var.set(round(cam['position'][1], 2))
            self.pos_z_var.set(round(cam['position'][2], 2))
            self.fov_var.set(cam['fov'])
    
    def update_selected_camera(self):
        """更新选中的相机参数"""
        if 0 <= self.selected_camera_index < len(self.config.cameras):
            try:
                position = np.array([self.pos_x_var.get(), self.pos_y_var.get(), self.pos_z_var.get()])
                
                # 计算朝向物体中心的方向
                look_at = self.config.object_center - position
                if np.linalg.norm(look_at) > 0:
                    look_at = look_at / np.linalg.norm(look_at)
                else:
                    look_at = np.array([0, 0, -1])  # 默认向下
                
                self.config.update_camera(
                    self.selected_camera_index,
                    position=position,
                    look_at=look_at,
                    fov=self.fov_var.get(),
                    name=self.name_var.get()
                )
                
                self.update_camera_list()
                self.update_3d_plot()
                
            except Exception as e:
                messagebox.showerror("错误", f"更新相机参数时出错: {e}")
    
    def add_camera_dialog(self):
        """添加相机对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("添加相机")
        dialog.geometry("300x250")
        dialog.resizable(False, False)
        
        # 相机名称
        ttk.Label(dialog, text="相机名称:").pack(pady=5)
        name_var = tk.StringVar(value=f"Camera_{len(self.config.cameras)+1}")
        ttk.Entry(dialog, textvariable=name_var).pack(pady=5)
        
        # 距离
        ttk.Label(dialog, text="距离物体中心:").pack(pady=5)
        distance_var = tk.DoubleVar(value=self.config.object_size * 1.5 if self.config.object_size > 0 else 100)
        ttk.Entry(dialog, textvariable=distance_var).pack(pady=5)
        
        # 角度
        ttk.Label(dialog, text="水平角度 (度):").pack(pady=5)
        angle_var = tk.DoubleVar(value=0)
        ttk.Entry(dialog, textvariable=angle_var).pack(pady=5)
        
        # 仰角
        ttk.Label(dialog, text="仰角 (度):").pack(pady=5)
        elevation_var = tk.DoubleVar(value=0)
        ttk.Entry(dialog, textvariable=elevation_var).pack(pady=5)
        
        def add_camera():
            try:
                angle_rad = np.radians(angle_var.get())
                elevation_rad = np.radians(elevation_var.get())
                distance = distance_var.get()
                center = self.config.object_center
                
                x = center[0] + distance * np.cos(elevation_rad) * np.cos(angle_rad)
                y = center[1] + distance * np.cos(elevation_rad) * np.sin(angle_rad)
                z = center[2] + distance * np.sin(elevation_rad)
                
                position = np.array([x, y, z])
                look_at = center - position
                if np.linalg.norm(look_at) > 0:
                    look_at = look_at / np.linalg.norm(look_at)
                else:
                    look_at = np.array([0, 0, -1])
                
                self.config.add_camera(position, look_at, 50.0, name_var.get())
                self.update_camera_list()
                self.update_3d_plot()
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("错误", f"添加相机时出错: {e}")
        
        ttk.Button(dialog, text="添加", command=add_camera).pack(pady=15)
        ttk.Button(dialog, text="取消", command=dialog.destroy).pack()
    
    def delete_selected_camera(self):
        """删除选中的相机"""
        if 0 <= self.selected_camera_index < len(self.config.cameras):
            self.config.remove_camera(self.selected_camera_index)
            self.selected_camera_index = -1
            self.update_camera_list()
            self.update_3d_plot()
    
    def copy_selected_camera(self):
        """复制选中的相机"""
        if 0 <= self.selected_camera_index < len(self.config.cameras):
            cam = self.config.cameras[self.selected_camera_index]
            new_name = f"{cam['name']}_copy"
            # 稍微偏移位置
            offset = self.config.object_size * 0.1 if self.config.object_size > 0 else 10
            new_position = cam['position'] + np.array([offset, offset, 0])
            self.config.add_camera(new_position, cam['look_at'], cam['fov'], new_name)
            self.update_camera_list()
            self.update_3d_plot()
    
    def add_circular_cameras(self, num_cameras: int):
        """添加环绕相机"""
        center = self.config.object_center
        radius = self.config.object_size * 1.5 if self.config.object_size > 0 else 100
        
        for i in range(num_cameras):
            angle = (2 * np.pi * i) / num_cameras
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + radius * 0.3  # 稍微仰视
            
            position = np.array([x, y, z])
            look_at = center - position
            if np.linalg.norm(look_at) > 0:
                look_at = look_at / np.linalg.norm(look_at)
            else:
                look_at = np.array([0, 0, -1])
            
            name = f"Circle_{i+1}_{int(np.degrees(angle))}°"
            self.config.add_camera(position, look_at, 50.0, name)
        
        self.update_camera_list()
        self.update_3d_plot()
    
    def add_four_direction_cameras(self):
        """添加前后左右4个相机"""
        center = self.config.object_center
        radius = self.config.object_size * 1.8 if self.config.object_size > 0 else 120
        elevation_height = self.config.object_size * 0.2 if self.config.object_size > 0 else 20
        
        # 定义四个方向：前、后、左、右
        directions = [
            {"name": "前方", "angle": 0},      # 前方 (0度)
            {"name": "右方", "angle": 90},     # 右方 (90度) 
            {"name": "后方", "angle": 180},    # 后方 (180度)
            {"name": "左方", "angle": 270}     # 左方 (270度)
        ]
        
        for direction in directions:
            angle_rad = np.radians(direction["angle"])
            
            # 计算相机位置 (稍微仰视)
            x = center[0] + radius * np.cos(angle_rad)
            y = center[1] + radius * np.sin(angle_rad)
            z = center[2] + elevation_height
            
            position = np.array([x, y, z])
            
            # 计算朝向物体中心的方向
            look_at = center - position
            if np.linalg.norm(look_at) > 0:
                look_at = look_at / np.linalg.norm(look_at)
            else:
                look_at = np.array([0, 0, -1])
            
            name = f"{direction['name']}相机"
            self.config.add_camera(position, look_at, 50.0, name)
        
        self.update_camera_list()
        self.update_3d_plot()
        
        messagebox.showinfo("成功", "已添加前后左右4个相机！")
    
    def clear_all_cameras(self):
        """清除所有相机"""
        if messagebox.askyesno("确认", "确定要删除所有相机吗？"):
            self.config.cameras = []
            self.selected_camera_index = -1
            self.update_camera_list()
            self.update_3d_plot()
    
    def load_stl_dialog(self):
        """加载STL文件对话框"""
        filename = filedialog.askopenfilename(
            title="选择STL文件",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )
        if filename:
            self.stl_file = filename
            self.load_stl_file(filename)
            self.update_camera_list()
            self.update_3d_plot()
    
    def save_config_dialog(self):
        """保存配置对话框"""
        filename = filedialog.asksaveasfilename(
            title="保存相机配置",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config.save_config(filename)
                messagebox.showinfo("成功", f"相机配置已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置时出错: {e}")
    
    def load_config_dialog(self):
        """加载配置对话框"""
        filename = filedialog.askopenfilename(
            title="加载相机配置",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config.load_config(filename)
                self.selected_camera_index = -1
                self.update_camera_list()
                self.update_3d_plot()
                messagebox.showinfo("成功", f"相机配置已从: {filename} 加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置时出错: {e}")
    
    def generate_pointcloud(self):
        """生成多角度点云 - 使用界面中设置的相机"""
        if not self.mesh or not self.config.cameras:
            messagebox.showwarning("警告", "请先加载STL文件并设置相机")
            return
        
        try:
            print(f"\n=== 使用 {len(self.config.cameras)} 个自定义相机生成点云 ===")
            print("1. 加载STL网格...")
            
            # 加载网格
            mesh = copy.deepcopy(self.mesh)
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            print(f"   网格顶点数: {len(vertices):,}")
            print(f"   网格三角面数: {len(triangles):,}")
            print(f"   物体中心: {self.config.object_center}")
            
            # 生成采样点
            num_sample_points = 100000
            print(f"2. 生成 {num_sample_points:,} 个采样点...")
            sampled_mesh = mesh.sample_points_uniformly(number_of_points=num_sample_points)
            sample_points = np.asarray(sampled_mesh.points)
            
            # 使用自定义相机位置
            print(f"3. 使用 {len(self.config.cameras)} 个自定义相机...")
            for i, cam in enumerate(self.config.cameras):
                print(f"   相机 {i+1}: {cam['name']} - 位置 {cam['position']}")
            
            # 创建KD树用于快速邻近搜索
            from sklearn.neighbors import KDTree
            kdtree = KDTree(sample_points)
            
            # 从每个相机视角获取可见点
            print("4. 从各个视角获取可见点...")
            all_visible_points = set()
            
            for i, cam in enumerate(self.config.cameras):
                print(f"   处理相机 {i+1}/{len(self.config.cameras)}: {cam['name']}...")
                
                camera_pos = cam['position']
                
                # 创建光线场景
                scene = o3d.t.geometry.RaycastingScene()
                mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                scene.add_triangles(mesh_t)
                
                # 计算从相机到每个采样点的光线
                rays_direction = sample_points - camera_pos
                rays_distance = np.linalg.norm(rays_direction, axis=1)
                rays_direction = rays_direction / rays_distance.reshape(-1, 1)
                
                # 创建光线起点（稍微偏移避免自相交）
                rays_origin = np.tile(camera_pos, (len(sample_points), 1)) + rays_direction * 0.001
                
                # 执行光线投射
                rays = np.concatenate([rays_origin, rays_direction], axis=1).astype(np.float32)
                ans = scene.cast_rays(o3d.core.Tensor(rays))
                
                # 获取命中距离
                hit_distances = ans['t_hit'].numpy()
                
                # 判断哪些点是可见的（光线没有被其他表面阻挡）
                visible_mask = (hit_distances >= (rays_distance - 0.01)) & (hit_distances != np.inf)
                visible_indices = np.where(visible_mask)[0]
                
                print(f"      可见点数: {len(visible_indices):,}")
                all_visible_points.update(visible_indices)
            
            # 合成最终视觉点云
            print("5. 合成最终视觉点云...")
            final_visible_indices = list(all_visible_points)
            final_visible_points = sample_points[final_visible_indices]
            
            print(f"\n=== 处理结果 ===")
            print(f"使用相机数: {len(self.config.cameras)}")
            print(f"初始采样点: {len(sample_points):,}")
            print(f"最终可见点: {len(final_visible_points):,} ({len(final_visible_points)/len(sample_points)*100:.1f}%)")
            
            # 创建点云对象
            final_pointcloud = o3d.geometry.PointCloud()
            final_pointcloud.points = o3d.utility.Vector3dVector(final_visible_points)
            final_pointcloud.paint_uniform_color([0, 1, 0])  # 绿色
            
            # 创建输出文件夹
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output_pointclouds_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 创建输出文件夹: {output_dir}")
            
            # 保存多种格式的结果
            print("6. 保存多种格式的点云文件...")
            base_filename = os.path.join(output_dir, "custom_camera_pointcloud")
            
            # 支持的点云格式
            formats = [
                {"ext": ".ply", "desc": "PLY格式 (Stanford Polygon Library)"},
                {"ext": ".pcd", "desc": "PCD格式 (Point Cloud Data)"},
                {"ext": ".xyz", "desc": "XYZ格式 (ASCII坐标)"},
                {"ext": ".pts", "desc": "PTS格式 (Points)"}
            ]
            
            saved_files = []
            for fmt in formats:
                output_file = base_filename + fmt["ext"]
                try:
                    success = o3d.io.write_point_cloud(output_file, final_pointcloud)
                    if success:
                        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                        print(f"   ✅ {fmt['desc']}: {output_file} ({file_size:.2f} MB)")
                        saved_files.append(output_file)
                    else:
                        print(f"   ❌ 保存{fmt['desc']}失败")
                except Exception as e:
                    print(f"   ❌ 保存{fmt['desc']}时出错: {e}")
            
            # 尝试重建网格并保存为STL
            print("   🔄 尝试重建网格并保存STL格式...")
            try:
                # 估计法向量
                final_pointcloud.estimate_normals()
                
                # 使用泊松重建
                mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    final_pointcloud, depth=8
                )
                
                # 保存STL网格
                stl_file = base_filename + ".stl"
                success = o3d.io.write_triangle_mesh(stl_file, mesh_poisson)
                if success:
                    file_size = os.path.getsize(stl_file) / (1024 * 1024)  # MB
                    print(f"   ✅ STL格式 (重建网格): {stl_file} ({file_size:.2f} MB)")
                    saved_files.append(stl_file)
                else:
                    print("   ❌ STL网格保存失败")
                    
            except Exception as e:
                print(f"   ❌ STL网格重建失败: {e}")
            
            # 保存原始采样点云（对比用）
            print("   💾 同时保存原始采样点云...")
            original_cloud = o3d.geometry.PointCloud()
            original_cloud.points = o3d.utility.Vector3dVector(sample_points)
            original_cloud.paint_uniform_color([1, 0, 0])  # 红色
            
            original_base = os.path.join(output_dir, "original_sampled_pointcloud")
            for fmt in formats[:2]:  # 只保存PLY和PCD格式
                original_file = original_base + fmt["ext"]
                try:
                    success = o3d.io.write_point_cloud(original_file, original_cloud)
                    if success:
                        file_size = os.path.getsize(original_file) / (1024 * 1024)  # MB
                        print(f"   📁 原始{fmt['desc']}: {original_file} ({file_size:.2f} MB)")
                        saved_files.append(original_file)
                except Exception as e:
                    print(f"   ❌ 保存原始{fmt['desc']}时出错: {e}")
            
            # 保存相机配置
            try:
                config_file = os.path.join(output_dir, "camera_config.json")
                self.config.save_config(config_file)
                saved_files.append(config_file)
                print(f"   📋 相机配置: {os.path.basename(config_file)}")
            except Exception as e:
                print(f"   ❌ 保存相机配置失败: {e}")
            
            # 生成README文件
            try:
                readme_file = os.path.join(output_dir, "README.txt")
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write("=== 自定义相机视觉点云输出 ===\n\n")
                    f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"使用相机数: {len(self.config.cameras)}\n")
                    f.write(f"初始采样点: {len(sample_points):,}\n")
                    f.write(f"最终可见点: {len(final_visible_points):,} ({len(final_visible_points)/len(sample_points)*100:.1f}%)\n\n")
                    
                    f.write("相机配置:\n")
                    for i, cam in enumerate(self.config.cameras, 1):
                        f.write(f"  {i}. {cam['name']}\n")
                        f.write(f"     位置: {cam['position']}\n")
                        f.write(f"     朝向: {cam['look_at']}\n")
                        f.write(f"     视场角: {cam['fov']}°\n\n")
                    
                    f.write("文件说明:\n")
                    f.write("- custom_camera_pointcloud.*: 经过4个相机视觉过滤的点云\n")
                    f.write("- original_sampled_pointcloud.*: 原始均匀采样点云（对比用）\n")
                    f.write("- camera_config.json: 相机配置文件\n")
                    if any('.stl' in f for f in saved_files):
                        f.write("- custom_camera_pointcloud.stl: 重建的网格模型\n")
                
                saved_files.append(readme_file)
                print(f"   📄 说明文档: {os.path.basename(readme_file)}")
            except Exception as e:
                print(f"   ❌ 生成README失败: {e}")
            
            print(f"\n📂 共保存了 {len(saved_files)} 个文件到文件夹: {output_dir}")
            
            # 可视化结果
            print("7. 可视化结果...")
            
            # 先显示原始采样点云
            original_cloud = o3d.geometry.PointCloud()
            original_cloud.points = o3d.utility.Vector3dVector(sample_points)
            original_cloud.paint_uniform_color([1, 0, 0])  # 红色
            
            print("    (显示原始采样点云 - 红色)")
            o3d.visualization.draw_geometries([original_cloud],
                                            window_name="原始采样点云",
                                            width=1000, height=700)
            
            # 再显示自定义相机视觉点云
            print("    (显示自定义相机视觉点云 - 绿色)")
            o3d.visualization.draw_geometries([final_pointcloud],
                                            window_name="自定义相机视觉点云",
                                            width=1000, height=700)
            
            # 最后显示对比
            print("    (对比显示: 红=原始, 绿=视觉可见)")
            o3d.visualization.draw_geometries([original_cloud, final_pointcloud],
                                            window_name="对比显示: 红=原始 vs 绿=视觉可见",
                                            width=1200, height=800)
            
            # 显示所有保存的文件列表
            for i, file in enumerate(saved_files, 1):
                file_size = os.path.getsize(file) / (1024 * 1024)  # MB
                print(f"   {i}. {file} ({file_size:.2f} MB)")
            
            print("=== 自定义相机点云生成完成 ===\n")
            
            # 准备消息框内容
            file_list = "\n".join([f"• {os.path.basename(f)}" for f in saved_files])
            messagebox.showinfo("完成", 
                f"点云生成完成！\n"
                f"使用了 {len(self.config.cameras)} 个相机\n"
                f"最终可见点: {len(final_visible_points):,} 个\n\n"
                f"输出文件夹: {output_dir}\n"
                f"保存的文件 ({len(saved_files)} 个):\n{file_list}")
            
        except Exception as e:
            print(f"❌ 生成点云时出错: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"生成点云时出错: {e}")
    
    def run(self):
        """运行编辑器"""
        self.root.mainloop()


def main():
    """主函数"""
    print("🎥 交互式相机编辑器")
    print("启动图形界面...")
    
    # 检查是否有默认STL文件
    default_stl = "lixun.STL"
    if os.path.exists(default_stl):
        editor = InteractiveCameraEditor(default_stl)
    else:
        editor = InteractiveCameraEditor()
    
    editor.run()


if __name__ == "__main__":
    main() 