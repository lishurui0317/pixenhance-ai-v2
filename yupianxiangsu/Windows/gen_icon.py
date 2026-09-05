import os
import math
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def generate_mj_3d_glass_icon(output_ico="app_icon.ico", output_png="app_icon.png"):
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # 1. 苹果/MJ 风格：底层大面积软弥散投影 (Soft Ambient Shadow)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow)
    s_draw.rounded_rectangle([48, 56, size - 48, size - 40], radius=110, fill=(5, 8, 20, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=32))
    img.alpha_composite(shadow)

    # 2. 哑光钛空灰/墨蓝微光底座 (Dark Slate Matte Base)
    base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(base)
    icon_box = [48, 48, size - 48, size - 48]
    
    for y in range(48, size - 48):
        ratio = (y - 48) / (size - 96)
        r = int(18 + (10 - 18) * ratio)
        g = int(22 + (14 - 22) * ratio)
        b = int(35 + (25 - 35) * ratio)
        b_draw.line([(48, y), (size - 48, y)], fill=(r, g, b, 255))
        
    mask = Image.new("L", (size, size), 0)
    m_draw = ImageDraw.Draw(mask)
    m_draw.rounded_rectangle(icon_box, radius=110, fill=255)
    base.putalpha(mask)
    img.alpha_composite(base)

    # 3. 3D 磨砂玻璃层 A：下沉背景折射光斑 (Refracted Backdrop Light)
    backdrop_light = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bl_draw = ImageDraw.Draw(backdrop_light)
    # 虹彩光谱渐变球 (Soft Spectrum Sphere)
    bl_draw.ellipse([140, 120, 360, 340], fill=(56, 189, 248, 140)) # 钴蓝
    bl_draw.ellipse([200, 180, 380, 360], fill=(168, 85, 247, 120)) # 紫罗兰
    bl_draw.ellipse([160, 220, 320, 380], fill=(236, 72, 153, 90))  # 玫瑰品红
    backdrop_light = backdrop_light.filter(ImageFilter.GaussianBlur(radius=40))
    img.alpha_composite(backdrop_light)

    # 4. 3D 磨砂玻璃层 B：核心悬浮凸透镜 (Floating Glass Convex Lens)
    # 模拟磨砂玻璃板主体 (Glassmorphic Plate)
    glass_plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gp_draw = ImageDraw.Draw(glass_plate)
    glass_box = [150, 150, 362, 362]
    
    # 玻璃板半透明填充 (Frosted Glass Fill)
    gp_draw.rounded_rectangle(glass_box, radius=70, fill=(255, 255, 255, 25))
    
    # 高斯模糊模拟磨砂核 (Frost Core Blur)
    frost_core = glass_plate.filter(ImageFilter.GaussianBlur(radius=8))
    img.alpha_composite(frost_core)

    # 5. 3D 物理光影：全反射边缘高光 (Specular Edge & Chromatic Rim Light)
    rim_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    r_draw = ImageDraw.Draw(rim_layer)
    
    # 顶部入射强光边 (Top Left Specular Highlight)
    r_draw.rounded_rectangle(glass_box, radius=70, outline=(255, 255, 255, 180), width=3)
    
    # 内侧次级折射反光环 (Inner Secondary Refraction)
    inner_glass_box = [153, 153, 359, 359]
    r_draw.rounded_rectangle(inner_glass_box, radius=67, outline=(255, 255, 255, 60), width=1)
    
    # 对角线折射光辉 (Diagonal Refractive Sweep)
    r_draw.line([(180, 160), (330, 310)], fill=(255, 255, 255, 45), width=18)
    r_draw.line([(210, 160), (340, 290)], fill=(255, 255, 255, 25), width=8)

    # 蒙版裁剪，确保高光不溢出透镜
    glass_mask = Image.new("L", (size, size), 0)
    gm_draw = ImageDraw.Draw(glass_mask)
    gm_draw.rounded_rectangle(glass_box, radius=70, fill=255)
    
    rim_masked = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rim_masked.paste(rim_layer, (0, 0), mask=glass_mask)
    img.alpha_composite(rim_masked)

    # 6. 外框顶部 Apple 边缘微光 (App Icon Top Bevel Highlight)
    outer_bevel = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ob_draw = ImageDraw.Draw(outer_bevel)
    ob_draw.rounded_rectangle(icon_box, radius=110, outline=(255, 255, 255, 50), width=2)
    img.alpha_composite(outer_bevel)

    # 7. 导出可直接预览、不报错的高清 ICO / PNG
    img.save(output_png, format="PNG")
    icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(output_ico, format="ICO", sizes=icon_sizes, bitmap_format="png")
    
    print(f"3D 磨砂玻璃透镜图标渲染完成：\n - ICO: {os.path.abspath(output_ico)}\n - PNG: {os.path.abspath(output_png)}")

if __name__ == "__main__":
    generate_mj_3d_glass_icon()
