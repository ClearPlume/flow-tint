import math
import random


def rgb_to_oklch(r, g, b):
    """
    将RGB颜色转换为OKLCh颜色空间（面向训练的整数化格式）

    OKLCh是一个感知均匀的颜色空间，更适合颜色计算和插值：
    - L (Lightness): 明度，0-100
    - C (Chroma): 彩度，0-0.4左右
    - h (Hue): 色相，0-360度

    参数:
        r, g, b: RGB值，范围0-255

    返回:
        tuple: (L, C, h) - 整数化的OKLCh值
            - L: 0-100000 (原值×1000)
            - C: 0-400000 (原值×1000)
            - h: 0-360000 (原值×1000)

    转换链路: RGB → Linear RGB → XYZ → OKLab → OKLCh → 整数化
    """

    # === 第一步：RGB标准化与线性化 ===
    # 将0-255的RGB值转换为0-1范围，并进行gamma校正
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    def srgb_to_linear(val):
        """sRGB gamma校正：移除显示器的非线性特性"""
        return val / 12.92 if val <= 0.04045 else pow((val + 0.055) / 1.055, 2.4)

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # === 第二步：线性RGB转XYZ色彩空间 ===
    # 使用D65标准光源的转换矩阵
    x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    # === 第三步：XYZ转OKLab ===
    # OKLab是感知均匀的色彩空间，先转换为中间锥体响应
    l_ = pow(0.8189330101 * x + 0.3618667424 * y - 0.1288597137 * z, 1 / 3)
    m_ = pow(0.0329845436 * x + 0.9293118715 * y + 0.0361456387 * z, 1 / 3)
    s_ = pow(0.0482003018 * x + 0.2643662691 * y + 0.6338517070 * z, 1 / 3)

    # 计算OKLab的Lab值
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_lab = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    # === 第四步：OKLab转OKLCh（极坐标形式）===
    # 将直角坐标(a,b)转换为极坐标(C,h)，更符合人类对颜色的感知
    C = math.sqrt(a * a + b_lab * b_lab)  # 彩度：距离原点的距离
    h = math.atan2(b_lab, a) * 180 / math.pi  # 色相：与a轴的夹角
    if h < 0:
        h += 360  # 确保色相在0-360度范围内

    # === 第五步：精度控制与整数化 ===
    # 保留3位小数精度，然后乘以1000转为整数（便于模型学习）
    return (
        round(L * 1000),  # 明度：0-100000
        round(C * 1000),  # 彩度：0-400000
        round(h * 1000),  # 色相：0-360000
    )


def add_fluctuation(value, fluctuation_percent=0.1):
    """
    对数值添加指定百分比的随机波动

    Args:
        value: 原始数值
        fluctuation_percent: 波动百分比，默认0.1（即10%）

    Returns:
        波动后的数值，保留3位小数
    """
    fluctuation_range = abs(value) * fluctuation_percent
    min_val = value - fluctuation_range
    max_val = value + fluctuation_range

    # 对于接近0的小数，确保不会改变正负号
    if value > 0 > min_val:
        min_val = 0.001
    elif value < 0 < max_val:
        max_val = -0.001

    result = random.uniform(min_val, max_val)
    return round(result, 3)
