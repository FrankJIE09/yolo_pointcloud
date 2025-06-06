import open3d as o3d
import argparse
import os

def stl_to_dense_surface_pcd(stl_path, pcd_out_path, num_points=100000):
    if not os.path.exists(stl_path):
        print(f"输入STL文件不存在: {stl_path}")
        return
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if not mesh.has_vertices():
        print(f"STL文件无有效三角面: {stl_path}")
        return
    print(f"采样表面点云: {num_points} 点 ...")
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)
    print(f"采样完成，点数: {len(pcd.points)}")
    o3d.io.write_point_cloud(pcd_out_path, pcd)
    print(f"已保存为PCD模板: {pcd_out_path}")
    print("正在弹窗展示采样点云，请关闭窗口继续...")
    o3d.visualization.draw_geometries([pcd], window_name="采样表面点云预览")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将STL文件表面采样为高密度点云PCD模板")
    parser.add_argument("--stl", type=str, default="lixun.STL", help="输入STL文件路径")
    parser.add_argument("--out", type=str, default="pcd_class_templates/lixun_template.pcd", help="输出PCD文件路径")
    parser.add_argument("--num_points", type=int, default=10000, help="采样点数，默认10万")
    args = parser.parse_args()
    stl_to_dense_surface_pcd(args.stl, args.out, args.num_points) 