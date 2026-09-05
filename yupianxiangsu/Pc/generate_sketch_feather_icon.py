#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sketch_feather_icon.py
- 512x512 手绘铅笔素描风格羽毛笔 Feather Quill
- 输出 PNG + 多尺寸 ICO（16/32/48/128/256）全 PNG 编码
- 暗哑光底座 + 暖白铅笔排线 + 羽毛羽枝细节 + 笔尖高光
- 技法：大量 1px 短笔触 + 随机抖动 + 阴影侧密集排线 + 纸张噪点
"""

import math
import os
import random
from PIL import Image, ImageDraw, ImageFilter

random.seed(11)

# ---------------------------------------------------------------------------
# 调色板：深哑光底 + 暖白铅笔
# ---------------------------------------------------------------------------
SIZE = 512
ICON_BOX = (44, 44, SIZE - 44, SIZE - 44)
RADIUS = 110

BG_TOP = (28, 28, 34)         # 顶部偏冷灰
BG_BOT = (18, 18, 24)         # 底部更深
PAPER = (255, 250, 240)       # 暖纸白

PENCIL_LIGHT = (238, 232, 218)   # 亮部 / 高光
PENCIL_MID = (190, 182, 168)     # 中调
PENCIL_DARK = (120, 114, 102)    # 暗部
PENCIL_SHADOW = (78, 74, 66)     # 重影
PENCIL_DEEP = (52, 48, 42)       # 极深
INK = (38, 36, 42)               # 笔尖深色

# ---------------------------------------------------------------------------
# 中心曲线：nib → 羽毛尖（cubic bezier）
# ---------------------------------------------------------------------------
P0 = (140, 412)    # nib 起点
P1 = (210, 360)
P2 = (305, 195)
P3 = (392, 100)    # 羽毛尖


def cubic(p0, p1, p2, p3, t):
    u = 1 - t
    x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
    y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
    return (x, y)


def ipt(p):
    """浮点坐标 → 整数坐标（Pillow 12 强制）"""
    return (int(round(p[0])), int(round(p[1])))


def tangent_at(pts, i):
    a = pts[max(0, i - 1)]
    b = pts[min(len(pts) - 1, i + 1)]
    return (b[0] - a[0], b[1] - a[1])


def normal(t):
    tx, ty = t
    L = math.hypot(tx, ty) or 1
    return (-ty / L, tx / L)


def jitter(amt):
    return (random.random() - 0.5) * 2 * amt


def round_corners(img: Image.Image, box, radius: int) -> Image.Image:
    """用 mask 切圆角（统一外框）"""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask=mask)
    return out


# ---------------------------------------------------------------------------
# 1. 底座：暗哑光渐变 + 纸张噪点 + 顶部微光
# ---------------------------------------------------------------------------
def draw_base(size: int) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # 纵向渐变
    grad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        r = y / size
        cr = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * r)
        cg = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * r)
        cb = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * r)
        gd.line([(0, y), (size, y)], fill=(cr, cg, cb, 255))

    # 纸张噪点
    noise = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    nd = ImageDraw.Draw(noise)
    for _ in range(4500):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        a = random.randint(8, 28)
        nd.point((x, y), fill=(255, 255, 255, a))
    noise = noise.filter(ImageFilter.GaussianBlur(radius=0.3))

    base_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base_layer.paste(grad, (0, 0))
    base_layer.alpha_composite(noise)
    base_layer = round_corners(base_layer, ICON_BOX, RADIUS)
    canvas.alpha_composite(base_layer)

    # 顶部 1px 微光描边
    bevel = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(bevel).rounded_rectangle(
        ICON_BOX, radius=RADIUS, outline=(255, 255, 255, 38), width=2
    )
    canvas.alpha_composite(bevel)
    return canvas


# ---------------------------------------------------------------------------
# 2. 笔身下方软投影
# ---------------------------------------------------------------------------
def draw_shadow(size: int) -> Image.Image:
    sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.ellipse([150, 425, 400, 470], fill=(0, 0, 0, 100))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=22))
    sh = round_corners(sh, ICON_BOX, RADIUS)
    return sh


# ---------------------------------------------------------------------------
# 3. 羽毛笔主图：shaft + vane + nib + hatching + 高光
# ---------------------------------------------------------------------------
def draw_quill(size: int, N: int = 240) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pts = [cubic(P0, P1, P2, P3, i / N) for i in range(N + 1)]
    d = ImageDraw.Draw(canvas)

    # 区段切分
    shaft_end = int(0.18 * N)            # 笔杆区段末端
    barb_start = int(0.18 * N)           # 羽毛根
    barb_end = int(0.94 * N)             # 羽毛尖散羽起点

    # ---------- A. 阴影侧密集排线 ----------
    # 在"暗面"（相对羽枝朝上为基准，约定左侧为暗面）打密集斜线
    for i in range(barb_start, barb_end):
        t = i / N
        ft = (t - 0.18) / 0.76
        envelope = abs(math.sin(math.pi * ft)) ** 0.55
        side_len = envelope * 130

        p = pts[i]
        t_vec = tangent_at(pts, i)
        nx, ny = normal(t_vec)

        for j in range(7):
            offset = (j / 7) * side_len
            sx = p[0] + nx * offset
            sy = p[1] + ny * offset
            tangent_angle = math.atan2(t_vec[1], t_vec[0])
            hatch_angle = tangent_angle + (random.random() - 0.5) * 0.35
            length = 4 + random.random() * 6
            ex = sx + math.cos(hatch_angle) * length
            ey = sy + math.sin(hatch_angle) * length
            color = random.choice([PENCIL_DARK, PENCIL_SHADOW, PENCIL_DEEP, PENCIL_SHADOW])
            d.line([ipt((sx, sy)), ipt((ex, ey))], fill=color, width=1)

    # ---------- B. 羽枝 ----------
    for i in range(barb_start, barb_end):
        t = i / N
        ft = (t - 0.18) / 0.76
        envelope = abs(math.sin(math.pi * ft)) ** 0.55
        side_len = envelope * 150

        p = pts[i]
        t_vec = tangent_at(pts, i)
        tlen = math.hypot(t_vec[0], t_vec[1]) or 1
        ux, uy = t_vec[0] / tlen, t_vec[1] / tlen
        nx, ny = normal(t_vec)

        for side in (+1, -1):
            for j in range(4):
                jt = j / 4
                # 羽枝长度: 内短外长
                barb_len = side_len * (0.45 + 0.55 * (0.3 + 0.7 * jt)) * (0.85 + 0.3 * random.random())
                # 羽枝略朝尖端（bias 越小越斜）
                bias = 0.22
                ex = p[0] + side * nx * barb_len * (1 - bias) + ux * barb_len * bias + jitter(0.9)
                ey = p[1] + side * ny * barb_len * (1 - bias) + uy * barb_len * bias + jitter(0.9)
                if side > 0:
                    color = random.choice([PENCIL_LIGHT, PENCIL_LIGHT, PENCIL_MID, PENCIL_MID, PENCIL_DARK])
                else:
                    # 暗面羽枝偏深
                    color = random.choice([PENCIL_MID, PENCIL_DARK, PENCIL_DARK, PENCIL_SHADOW])
                d.line([ipt(p), ipt((ex, ey))], fill=color, width=1)

    # ---------- C. 羽毛尖端散羽（plumage wisps） ----------
    for i in range(barb_end, len(pts)):
        p = pts[i]
        t_vec = tangent_at(pts, i)
        nx, ny = normal(t_vec)
        for k in range(2):
            side = random.choice((-1, 1))
            length = random.uniform(30, 95)
            ex = p[0] + side * nx * length + jitter(3)
            ey = p[1] + side * ny * length + jitter(3)
            color = random.choice([PENCIL_LIGHT, PENCIL_MID])
            d.line([ipt(p), ipt((ex, ey))], fill=color, width=1)

    # ---------- D. 中轴羽轴 (rachis) — 比 shaft 略粗，贯穿全羽毛 ----------
    for i in range(barb_start, len(pts) - 1):
        p_a = pts[i]
        p_b = pts[i + 1]
        # 暗部底层
        d.line([ipt(p_a), ipt(p_b)], fill=PENCIL_SHADOW, width=5)
        # 中调
        d.line(
            [ipt((p_a[0] - 0.4, p_a[1] - 0.4)), ipt((p_b[0] - 0.4, p_b[1] - 0.4))],
            fill=PENCIL_MID, width=3,
        )
        # 亮部高光
        d.line(
            [ipt((p_a[0] - 1.0, p_a[1] - 1.0)), ipt((p_b[0] - 1.0, p_b[1] - 1.0))],
            fill=PENCIL_LIGHT, width=1,
        )

    # ---------- E. 笔杆 (shaft) ----------
    for i in range(shaft_end + 1):
        p_a = pts[i]
        p_b = pts[i + 1]
        # 暗部底层
        d.line([ipt(p_a), ipt(p_b)], fill=PENCIL_SHADOW, width=13)
        # 中调主色
        d.line(
            [ipt((p_a[0] - 0.6, p_a[1] - 0.6)), ipt((p_b[0] - 0.6, p_b[1] - 0.6))],
            fill=PENCIL_MID, width=9,
        )
        # 亮部高光
        d.line(
            [ipt((p_a[0] - 1.6, p_a[1] - 1.6)), ipt((p_b[0] - 1.6, p_b[1] - 1.6))],
            fill=PENCIL_LIGHT, width=3,
        )

    # ---------- F. 笔尖 (nib) ----------
    dx, dy = P1[0] - P0[0], P1[1] - P0[1]
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L           # 笔杆方向
    px, py = -uy, ux                  # 垂直方向

    nib_len = 40
    nib_half_w = 11
    nib_tip = (P0[0] + ux * nib_len, P0[1] + uy * nib_len)
    base_l = (P0[0] + px * nib_half_w, P0[1] + py * nib_half_w)
    base_r = (P0[0] - px * nib_half_w, P0[1] - py * nib_half_w)

    # nib 主体：深色填充 + 高光描边
    d.polygon([ipt(base_l), ipt(nib_tip), ipt(base_r)], fill=INK, outline=PENCIL_LIGHT)

    # nib 中央狭缝
    slit_a = (P0[0] + ux * nib_len * 0.45, P0[1] + uy * nib_len * 0.45)
    slit_b = (P0[0] + ux * nib_len * 0.88, P0[1] + uy * nib_len * 0.88)
    d.line([ipt(slit_a), ipt(slit_b)], fill=PENCIL_LIGHT, width=1)

    # nib 中心墨孔
    hole = (P0[0] + ux * nib_len * 0.40, P0[1] + uy * nib_len * 0.40)
    d.ellipse([hole[0] - 2.5, hole[1] - 2.5, hole[0] + 2.5, hole[1] + 2.5], fill=PENCIL_LIGHT)

    # nib 高光斜线（沿笔杆方向）
    hl_a = (base_l[0] + ux * 4, base_l[1] + uy * 4)
    hl_b = (nib_tip[0] - ux * 5, nib_tip[1] - uy * 5)
    d.line([ipt(hl_a), ipt(hl_b)], fill=(255, 255, 255, 160), width=1)

    # ---------- G. 整体柔化（让笔触有手绘"绒"感） ----------
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))
    canvas = round_corners(canvas, ICON_BOX, RADIUS)
    return canvas


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(11)
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    img.alpha_composite(draw_base(SIZE))
    img.alpha_composite(draw_shadow(SIZE))
    img.alpha_composite(draw_quill(SIZE))

    out_png = "app_icon.png"
    out_ico = "app_icon.ico"
    img.save(out_png, format="PNG")

    icon_sizes = [(16, 16), (32, 32), (48, 48), (128, 128), (256, 256)]
    img.save(out_ico, format="ICO", sizes=icon_sizes, bitmap_format="png")
    print(f"✅ 素描羽毛笔图标已生成：\n - {os.path.abspath(out_png)}\n - {os.path.abspath(out_ico)}")


if __name__ == "__main__":
    main()
