#!/usr/bin/env python3
"""
多角度视觉点云生成器
模拟相机在多个角度拍摄物体，获取每个角度的可见点，然后组合成完整的视觉点云
"""

import open3d as o3d
import numpy as np
import copy
import time
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MultiViewVisualPointCloud:
    """多角度视觉点云生成器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def generate_camera_positions(
        self, 
        center: np.ndarray, 
        radius: float, 
        num_views: int = 8,
        elevation_angles: List[float] = None,
        full_sphere: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成相机位置和朝向
        
        Args:
            center: 物体中心
            radius: 相机距离物体的半径
            num_views: 水平方向视角数量
            elevation_angles: 仰角列表 (度)
            full_sphere: 是否生成全球面视角
            
        Returns:
            [(camera_position, look_at_direction), ...]
        """
        if elevation_angles is None:
            if full_sphere:
                elevation_angles = [-45, -20, 0, 20, 45, 70]  # 更多仰角
            else:
                elevation_angles = [0, 20, 45]  # 常用拍摄角度
        
        camera_positions = []
        
        for elevation in elevation_angles:
            elevation_rad = math.radians(elevation)
            
            for i in range(num_views):
                # 水平角度
                azimuth = (2 * math.pi * i) / num_views
                
                # 计算相机位置
                x = center[0] + radius * math.cos(elevation_rad) * math.cos(azimuth)
                y = center[1] + radius * math.cos(elevation_rad) * math.sin(azimuth)
                z = center[2] + radius * math.sin(elevation_rad)
                
                camera_pos = np.array([x, y, z])
                
                # 朝向物体中心
                look_at = center - camera_pos
                look_at = look_at / np.linalg.norm(look_at)
                
                camera_positions.append((camera_pos, look_at))
        
        return camera_positions
    
    def visualize_camera_setup(self, center: np.ndarray, camera_positions: List[Tuple[np.ndarray, np.ndarray]]):
        """可视化相机设置"""
        self.log("可视化相机位置设置...")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 画物体中心
        ax.scatter(center[0], center[1], center[2], color='red', s=100, label='物体中心')
        
        # 画相机位置
        camera_coords = np.array([pos for pos, _ in camera_positions])
        ax.scatter(camera_coords[:, 0], camera_coords[:, 1], camera_coords[:, 2], 
                  color='blue', s=50, label='相机位置')
        
        # 画视线方向
        for i, (cam_pos, look_at) in enumerate(camera_positions[:12]):  # 只显示前12个避免太乱
            end_pos = cam_pos + look_at * 0.3
            ax.plot([cam_pos[0], end_pos[0]], 
                   [cam_pos[1], end_pos[1]], 
                   [cam_pos[2], end_pos[2]], 'g--', alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')  
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'相机设置 ({len(camera_positions)}个视角)')
        plt.show()
    
    def get_visible_points_from_viewpoint(
        self, 
        mesh: o3d.geometry.TriangleMesh, 
        sample_points: np.ndarray,
        camera_pos: np.ndarray, 
        look_at: np.ndarray,
        fov: float = 60.0,
        max_distance: float = None
    ) -> np.ndarray:
        """
        从特定视角获取可见点
        
        Args:
            mesh: 原始网格
            sample_points: 采样点
            camera_pos: 相机位置
            look_at: 视线方向
            fov: 视场角 (度)
            max_distance: 最大可见距离
            
        Returns:
            可见点的索引
        """
        # 1. 视场角过滤
        to_points = sample_points - camera_pos
        distances = np.linalg.norm(to_points, axis=1)
        to_points_normalized = to_points / distances.reshape(-1, 1)
        
        # 计算与视线方向的夹角
        dot_products = np.dot(to_points_normalized, look_at)
        fov_rad = math.radians(fov / 2)  # 半视场角
        fov_mask = dot_products > math.cos(fov_rad)
        
        if max_distance is not None:
            distance_mask = distances < max_distance
            fov_mask = fov_mask & distance_mask
        
        # 2. 可见性检测 (射线投射)
        visible_indices = []
        candidate_indices = np.where(fov_mask)[0]
        
        # 创建RaycastingScene用于射线投射
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        
        # 批量射线投射
        rays_origin = np.tile(camera_pos, (len(candidate_indices), 1)).astype(np.float32)
        rays_direction = (sample_points[candidate_indices] - camera_pos).astype(np.float32)
        rays_direction = rays_direction / np.linalg.norm(rays_direction, axis=1, keepdims=True)
        
        # 执行射线投射
        ans = scene.cast_rays(o3d.core.Tensor(np.column_stack([rays_origin, rays_direction])))
        hit_distances = ans['t_hit'].numpy()
        
        # 检查哪些点是可见的
        point_distances = np.linalg.norm(sample_points[candidate_indices] - camera_pos, axis=1)
        
        # 如果射线击中的距离与点的距离接近，说明点是可见的
        tolerance = 0.01  # 允许的误差
        for i, idx in enumerate(candidate_indices):
            if not np.isinf(hit_distances[i]):
                if abs(hit_distances[i] - point_distances[i]) < tolerance:
                    visible_indices.append(idx)
        
        return np.array(visible_indices)
    
    def generate_multi_view_pointcloud(
        self,
        stl_file: str,
        output_file: str = None,
        num_sample_points: int = 100000,
        num_horizontal_views: int = 8,
        elevation_angles: List[float] = None,
        camera_distance_factor: float = 1.5,
        fov: float = 60.0,
        visualize_setup: bool = True,
        visualize_result: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        生成多角度视觉点云
        
        Args:
            stl_file: STL文件路径
            output_file: 输出文件路径
            num_sample_points: 初始采样点数
            num_horizontal_views: 水平方向视角数
            elevation_angles: 仰角列表
            camera_distance_factor: 相机距离因子
            fov: 视场角
            visualize_setup: 是否可视化相机设置
            visualize_result: 是否可视化结果
            
        Returns:
            合成的视觉点云
        """
        start_time = time.time()
        self.log(f"=== 多角度视觉点云生成: {stl_file} ===")
        
        # 1. 加载STL文件
        self.log("1. 加载STL网格...")
        mesh = o3d.io.read_triangle_mesh(stl_file)
        if not mesh.has_vertices():
            self.log("错误：无法加载STL文件")
            return None
        
        self.log(f"   网格顶点数: {len(mesh.vertices):,}")
        self.log(f"   网格三角面数: {len(mesh.triangles):,}")
        
        # 计算网格属性
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 2. 计算物体中心和尺寸
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)
        camera_distance = diagonal * camera_distance_factor
        
        self.log(f"   物体中心: {center}")
        self.log(f"   对角线长度: {diagonal:.3f}")
        self.log(f"   相机距离: {camera_distance:.3f}")
        
        # 3. 生成采样点
        self.log(f"2. 生成 {num_sample_points:,} 个采样点...")
        sample_pcd = mesh.sample_points_uniformly(number_of_points=num_sample_points)
        sample_points = np.asarray(sample_pcd.points)
        
        # 4. 生成相机位置
        self.log("3. 生成相机位置...")
        if elevation_angles is None:
            elevation_angles = [0, 20, 45, -20]  # 默认多个仰角
        
        camera_positions = self.generate_camera_positions(
            center=center,
            radius=camera_distance,
            num_views=num_horizontal_views,
            elevation_angles=elevation_angles
        )
        
        total_views = len(camera_positions)
        self.log(f"   生成 {total_views} 个视角")
        self.log(f"   水平视角: {num_horizontal_views}, 仰角: {elevation_angles}")
        
        # 5. 可视化相机设置
        if visualize_setup:
            self.visualize_camera_setup(center, camera_positions)
        
        # 6. 从每个视角获取可见点
        self.log("4. 从各个视角获取可见点...")
        all_visible_indices = set()
        view_stats = []
        
        for i, (camera_pos, look_at) in enumerate(camera_positions):
            self.log(f"   处理视角 {i+1}/{total_views}...")
            
            visible_indices = self.get_visible_points_from_viewpoint(
                mesh=mesh,
                sample_points=sample_points,
                camera_pos=camera_pos,
                look_at=look_at,
                fov=fov
            )
            
            self.log(f"      可见点数: {len(visible_indices):,}")
            view_stats.append(len(visible_indices))
            all_visible_indices.update(visible_indices)
        
        # 7. 合成最终点云
        self.log("5. 合成最终视觉点云...")
        final_indices = list(all_visible_indices)
        final_points = sample_points[final_indices]
        
        # 创建最终点云
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(final_points)
        
        # 计算法线
        bbox = final_pcd.get_axis_aligned_bounding_box()
        diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)
        radius_normal = diagonal / 50
        
        final_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        
        # 8. 统计和保存
        processing_time = time.time() - start_time
        
        self.log(f"\n=== 处理结果 ===")
        self.log(f"总视角数: {total_views}")
        self.log(f"初始采样点: {num_sample_points:,}")
        self.log(f"最终可见点: {len(final_points):,} ({len(final_points)/num_sample_points*100:.1f}%)")
        self.log(f"平均每视角可见点: {np.mean(view_stats):.0f}")
        self.log(f"处理时间: {processing_time:.2f}秒")
        
        # 9. 可视化结果
        if visualize_result:
            self.log("6. 可视化结果...")
            
            # 显示原始采样点云
            sample_pcd.paint_uniform_color([1, 0, 0])  # 红色
            self.log("    (显示原始采样点云 - 红色)")
            o3d.visualization.draw_geometries([sample_pcd], window_name="原始采样点云")
            
            # 显示视觉可见点云
            final_pcd.paint_uniform_color([0, 1, 0])  # 绿色
            self.log("    (显示多角度视觉点云 - 绿色)")
            o3d.visualization.draw_geometries([final_pcd], window_name="多角度视觉点云")
            
            # 对比显示
            sample_pcd_copy = copy.deepcopy(sample_pcd)
            final_pcd_copy = copy.deepcopy(final_pcd)
            sample_pcd_copy.paint_uniform_color([1, 0, 0])  # 红色原始
            final_pcd_copy.paint_uniform_color([0, 1, 0])   # 绿色可见
            self.log("    (对比显示: 红=原始, 绿=视觉可见)")
            o3d.visualization.draw_geometries([sample_pcd_copy, final_pcd_copy], 
                                            window_name="对比: 红=原始采样, 绿=视觉可见")
        
        # 10. 保存结果
        if output_file is None:
            import os
            base_name = os.path.splitext(stl_file)[0]
            output_file = f"{base_name}_multi_view_visual.ply"
        
        self.log(f"7. 保存到: {output_file}")
        o3d.io.write_point_cloud(output_file, final_pcd)
        
        self.log("=== 多角度视觉点云生成完成 ===\n")
        return final_pcd


def main():
    """主函数"""
    generator = MultiViewVisualPointCloud()
    
    # 配置参数
    stl_file = "lixun.STL"
    num_sample_points = 100000  # 增加采样点数以获得更好效果
    num_horizontal_views = 12   # 水平方向12个视角 (每30度一个)
    elevation_angles = [0, 15, 30, 45, -15, -30]  # 多个仰角
    
    print("🎥 多角度视觉点云生成器")
    print("=" * 60)
    print(f"STL文件: {stl_file}")
    print(f"采样点数: {num_sample_points:,}")
    print(f"水平视角: {num_horizontal_views} (每{360//num_horizontal_views}度)")
    print(f"仰角: {elevation_angles}")
    print(f"总视角数: {len(elevation_angles) * num_horizontal_views}")
    print("-" * 60)
    
    # 生成多角度视觉点云
    result = generator.generate_multi_view_pointcloud(
        stl_file=stl_file,
        num_sample_points=num_sample_points,
        num_horizontal_views=num_horizontal_views,
        elevation_angles=elevation_angles,
        camera_distance_factor=1.8,  # 相机距离稍远一些
        fov=50.0,  # 略小的视场角，更像真实相机
        visualize_setup=True,
        visualize_result=True
    )
    
    if result:
        print("\n🎉 多角度视觉点云生成完成！")
        print("生成的点云更真实地反映了基于视觉的点云获取过程")


if __name__ == "__main__":
    main() 