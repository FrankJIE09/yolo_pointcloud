#!/usr/bin/env python3
"""
修复matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import warnings

def find_chinese_fonts():
    """查找可用的中文字体"""
    print("🔍 查找系统中的中文字体...")
    
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        if ('CJK' in font_name or 'Zen Hei' in font_name or 
            'SimHei' in font_name or 'Microsoft YaHei' in font_name or
            'WenQuanYi' in font_name or 'Noto' in font_name):
            chinese_fonts.append(font_name)
    
    # 去重并排序
    chinese_fonts = sorted(list(set(chinese_fonts)))
    
    print(f"找到 {len(chinese_fonts)} 个中文字体:")
    for i, font in enumerate(chinese_fonts):
        print(f"  {i+1}. {font}")
    
    return chinese_fonts

def configure_chinese_font():
    """配置中文字体"""
    print("\n⚙️  配置中文字体...")
    
    # 查找可用字体
    chinese_fonts = find_chinese_fonts()
    
    if not chinese_fonts:
        print("❌ 未找到可用的中文字体！")
        return False
    
    # 按优先级设置字体
    priority_fonts = [
        'Noto Sans CJK SC',
        'WenQuanYi Zen Hei', 
        'Noto Sans CJK JP',
        'Noto Serif CJK SC',
        'Droid Sans Fallback'
    ]
    
    # 找到第一个可用的字体
    selected_font = None
    for font in priority_fonts:
        if font in chinese_fonts:
            selected_font = font
            break
    
    if not selected_font and chinese_fonts:
        selected_font = chinese_fonts[0]
    
    print(f"✅ 选择字体: {selected_font}")
    
    # 设置matplotlib参数
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    
    return True

def test_chinese_display():
    """测试中文显示效果"""
    print("\n🧪 测试中文显示...")
    
    # 配置字体
    if not configure_chinese_font():
        return
    
    # 创建测试图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 添加中文内容
    ax.set_title("相机编辑器 - 中文字体测试", fontsize=16)
    ax.set_xlabel("X 轴位置", fontsize=12)
    ax.set_ylabel("Y 轴位置", fontsize=12)
    
    # 绘制测试数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', label="正弦波", linewidth=2)
    
    # 添加中文标注
    ax.text(5, 0.8, "这是中文测试文本", fontsize=14, ha='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    try:
        plt.savefig("chinese_font_fixed.png", dpi=150, bbox_inches='tight')
        print("✅ 测试图片已保存: chinese_font_fixed.png")
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
    
    plt.close()

if __name__ == "__main__":
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    print("🚀 修复matplotlib中文字体显示问题")
    test_chinese_display()
    print("✅ 修复完成！") 