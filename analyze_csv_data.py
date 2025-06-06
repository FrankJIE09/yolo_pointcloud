import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.font_manager as fm

# 配置中文字体
def setup_chinese_font():
    # 尝试各种可能的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                    'Noto Sans CJK SC', 'Source Han Sans CN', 'AR PL UMing CN']
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 寻找可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"找到中文字体: {font}")
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
    else:
        print("未找到专门的中文字体，尝试使用系统默认字体")
        # 尝试使用 DejaVu Sans 的变体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans'] + plt.rcParams['font.sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print(f"当前字体设置: {plt.rcParams['font.sans-serif']}")

setup_chinese_font()

# 你的数据
data = [
    [2, -49.98916016770028, 74.95910354765701, 372.4529938456383, -3.5561021482341046, 0.6690003663053214, -0.11940397234529043],
    [3, -50.30633867591593, 74.73381280057609, 372.88391857955787, -1.6710865165327795, 1.2945021535781576, 0.4655641870138821],
    [4, 529.4947089837914, -227.7593658545801, 618.8115793131244, 21.83067438220691, -4.455129293452851, -32.18254691238069],
    [5, 529.7076418982842, -231.3744154902835, 619.4881700784049, 23.60432520119854, -4.83493736046979, -32.52685465477279],
    [6, 529.3359277673728, -228.201842317937, 618.7417312469877, 22.16806688853987, -4.126990741693104, -30.483550470070185],
    [7, 533.5018882258224, -234.87795564674528, 619.411146676967, 25.495911299136512, -1.8730561244424289, -25.951203974457588],
    [8, 533.0011084570126, -235.37298087157214, 619.3837022891736, 25.36214214317176, -1.7790389195576988, -26.361990266165748],
    [9, -49.152634103672845, 75.41672950401325, 372.63465283738753, -3.232323953733032, 0.9184719026841944, 0.33482516047139427]
]

# 转换为numpy数组
data_array = np.array(data)
indices = data_array[:, 0].astype(int)
pose_data = data_array[:, 1:]  # tx, ty, tz, rx, ry, rz

print("="*80)
print("                     重复性测试数据分析报告")
print("="*80)

# 1. 基础统计
print("\n【原始数据概览】")
print("检测序号:", indices.tolist())
print("数据维度:", pose_data.shape)

# 2. 使用K-means聚类分析平移数据
translation_data = pose_data[:, :3]  # tx, ty, tz
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(translation_data)

print(f"\n【K-means聚类分析（平移数据）】")
print("聚类结果:", clusters.tolist())

# 3. 分组分析
group1_indices = indices[clusters == 0]
group2_indices = indices[clusters == 1] 
group1_data = pose_data[clusters == 0]
group2_data = pose_data[clusters == 1]

print(f"\n【分组结果】")
print(f"第一组检测序号: {group1_indices.tolist()}")
print(f"第二组检测序号: {group2_indices.tolist()}")

# 4. 各组统计分析
def analyze_group(group_data, group_name):
    print(f"\n【{group_name} 统计分析】")
    print(f"样本数量: {len(group_data)}")
    
    mean = np.mean(group_data, axis=0)
    std = np.std(group_data, axis=0)
    min_vals = np.min(group_data, axis=0)
    max_vals = np.max(group_data, axis=0)
    
    print("                    tx(mm)        ty(mm)        tz(mm)        rx(°)         ry(°)         rz(°)")
    print(f"均值         {mean[0]:12.6f} {mean[1]:12.6f} {mean[2]:12.6f} {mean[3]:12.6f} {mean[4]:12.6f} {mean[5]:12.6f}")
    print(f"标准差       {std[0]:12.6f} {std[1]:12.6f} {std[2]:12.6f} {std[3]:12.6f} {std[4]:12.6f} {std[5]:12.6f}")
    print(f"最小值       {min_vals[0]:12.6f} {min_vals[1]:12.6f} {min_vals[2]:12.6f} {min_vals[3]:12.6f} {min_vals[4]:12.6f} {min_vals[5]:12.6f}")
    print(f"最大值       {max_vals[0]:12.6f} {max_vals[1]:12.6f} {max_vals[2]:12.6f} {max_vals[3]:12.6f} {max_vals[4]:12.6f} {max_vals[5]:12.6f}")
    
    # 计算重复性精度
    translation_distances = np.sqrt(np.sum((group_data[:, :3] - mean[:3])**2, axis=1))
    rotation_angles = np.sqrt(np.sum((group_data[:, 3:] - mean[3:])**2, axis=1))
    
    trans_repeatability = np.std(translation_distances)
    rot_repeatability = np.std(rotation_angles)
    
    print(f"平移重复定位精度 (1σ): {trans_repeatability:.6f} mm")
    print(f"旋转重复定位精度 (1σ): {rot_repeatability:.6f}°")
    
    return mean, std, trans_repeatability, rot_repeatability

group1_stats = analyze_group(group1_data, "第一组")
group2_stats = analyze_group(group2_data, "第二组")

# 5. 两组间差异分析
print(f"\n【两组间差异分析】")
diff_mean = np.abs(group1_stats[0] - group2_stats[0])
print("                    tx(mm)        ty(mm)        tz(mm)        rx(°)         ry(°)         rz(°)")
print(f"均值差异     {diff_mean[0]:12.6f} {diff_mean[1]:12.6f} {diff_mean[2]:12.6f} {diff_mean[3]:12.6f} {diff_mean[4]:12.6f} {diff_mean[5]:12.6f}")

translation_distance_between_groups = np.sqrt(np.sum(diff_mean[:3]**2))
rotation_distance_between_groups = np.sqrt(np.sum(diff_mean[3:]**2))

print(f"\n两组间3D平移距离: {translation_distance_between_groups:.6f} mm")
print(f"两组间3D旋转差异: {rotation_distance_between_groups:.6f}°")

# 6. 问题诊断
print(f"\n" + "="*80)
print("                           问题诊断")
print("="*80)

print("\n【结论】")
print("❌ 这NOT是同一个物体的重复性测试！")
print("✅ 数据显示检测到了两个不同位置/姿态的物体")

print(f"\n【证据】")
print(f"1. 平移差异: {translation_distance_between_groups:.1f}mm - 远超正常重复性误差")
print(f"2. 旋转差异: {rotation_distance_between_groups:.1f}° - 说明物体姿态完全不同")
print(f"3. 明显的聚类: 数据明确分为两组")

print(f"\n【可能原因】")
print("1. 🔄 物体在测试过程中被移动到了新位置")
print("2. 🎯 YOLO检测框跳跃，检测到了不同的目标区域")
print("3. 📷 相机位置或视角发生了变化")
print("4. 🚫 ICP配准失败，产生了错误的姿态估计")
print("5. 🔍 场景中存在多个相似物体，检测在它们之间跳跃")

print(f"\n【建议解决方案】")
print("1. 🔒 确保物体在整个测试过程中保持静止")
print("2. 📏 检查YOLO检测框的稳定性（可添加检测框可视化）")
print("3. 🎛️ 调整ICP参数，提高配准精度")
print("4. 📊 添加异常值检测，自动剔除明显错误的结果")
print("5. 🔍 在测试开始前确认只有一个目标物体在视野中")

# 7. 可视化
plt.figure(figsize=(15, 10))

# 平移数据3D散点图
ax1 = plt.subplot(2, 2, 1, projection='3d')
colors = ['red' if c == 0 else 'blue' for c in clusters]
ax1.scatter(translation_data[:, 0], translation_data[:, 1], translation_data[:, 2], c=colors, s=100)
ax1.set_xlabel('tx (mm)')
ax1.set_ylabel('ty (mm)')
ax1.set_zlabel('tz (mm)')
ax1.set_title('平移数据聚类（红色=组1，蓝色=组2）')

# 旋转数据3D散点图
ax2 = plt.subplot(2, 2, 2, projection='3d')
rotation_data = pose_data[:, 3:]
ax2.scatter(rotation_data[:, 0], rotation_data[:, 1], rotation_data[:, 2], c=colors, s=100)
ax2.set_xlabel('rx (°)')
ax2.set_ylabel('ry (°)')
ax2.set_zlabel('rz (°)')
ax2.set_title('旋转数据聚类（红色=组1，蓝色=组2）')

# 时间序列图
ax3 = plt.subplot(2, 1, 2)
ax3.plot(indices, translation_data[:, 0], 'o-', label='tx', markersize=8)
ax3.plot(indices, translation_data[:, 1], 's-', label='ty', markersize=8)
ax3.plot(indices, translation_data[:, 2], '^-', label='tz', markersize=8)
ax3.set_xlabel('检测序号')
ax3.set_ylabel('位置 (mm)')
ax3.set_title('平移数据时间序列（可见明显跳跃）')
ax3.legend()
ax3.grid(True)

# 标记异常跳跃点
for i in range(1, len(indices)):
    prev_pos = translation_data[i-1]
    curr_pos = translation_data[i]
    distance = np.sqrt(np.sum((curr_pos - prev_pos)**2))
    if distance > 100:  # 100mm阈值
        ax3.axvline(x=indices[i], color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax3.text(indices[i], ax3.get_ylim()[1]*0.9, f'跳跃\n{distance:.0f}mm', 
                ha='center', va='top', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('repeatability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n可视化图表已保存为: repeatability_analysis.png")
print("="*80) 