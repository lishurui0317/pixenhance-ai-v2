#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片批量超分辨率放大工具（v4.0 极简白 · 双引擎智能切换）
================================================================
核心特性
  - 双引擎无缝切换：Vulkan GPU（realesrgan-ncnn-vulkan.exe）→ ONNX CPU 保底
  - 极简白 UI（#F8F9FA）+ 纯黑按钮高亮 + 鼠标流体漩涡
  - 启动自动检测 MSVCP140.dll / onnxruntime，依赖缺失弹窗
  - Toast 气泡非模态提示（不阻塞 UI）
  - 拖放自动过滤非图片格式，弹窗告知用户
  - CPU 模式 + 图片 > 2048x2048 → 自动切片 + 气泡预警
  - config.json 持久化：last_opened_path / scale_factor / model_name
  - 严禁任何 C:\\Windows / C:\\Users 硬编码路径
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:  # pragma: no cover
    DND_AVAILABLE = False

# ONNX 推理 + 图像处理依赖（软依赖，缺失时降级）
try:
    import numpy as np
    NUMPY_OK = True
except ImportError:  # pragma: no cover
    NUMPY_OK = False

try:
    import onnxruntime as ort
    ONNX_RUNTIME_OK = True
except ImportError:  # pragma: no cover
    ONNX_RUNTIME_OK = False

try:
    from PIL import Image, ImageFilter
    PIL_OK = True
except ImportError:  # pragma: no cover
    PIL_OK = False


# ===========================================================================
# 常量
# ===========================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
APP_NAME = "图片无损画质增强大师"
APP_VERSION = "v4.0"
SCALE_OPTIONS = ("2x", "3x", "4x", "FHD", "2K", "4K")
# 目标档位（长边像素）。横向/竖向都按长边归一。
TARGET_PRESETS = {
    "FHD": 1920,
    "2K":  2560,
    "4K":  3840,
}
# 阈值：超过这个长边就不值得跑 Vulkan（直接 LANCZOS + 后处理，节省 GPU）。
# 经验值：~1700px 已经是 2.9MP，再 x4 会触 11.6MP，超过 2060 的甜蜜点。
VULKAN_SKIP_LONG_EDGE = 1700
# 切 tile 阈值：原图超过 1M 像素就走切 tile 模式（防 4K 原图跑 16K 爆显存）
# tile=128 后 Vulkan 内部每 tile 的中间 buffer 大幅缩小，显存峰值 ≤ 1.5GB
VULKAN_TILE_PIXEL_LIMIT = 1_000_000
# Vulkan tile 内部尺寸：-t 参数。128 是 RTX 2060 6GB 显存的安全上限
# （之前用 400 几乎吃光 6GB）
VULKAN_TILE_SIZE = 128
# 外层并发：只开 1 防止多张图叠加显存
MAX_WORKERS = 1
FILE_LIST_MAX_SHOW = 500

# ONNX 模式触发分辨率阈值
ONNX_RESOLUTION_WARN_PX = 2048

# Toast 默认参数
TOAST_DEFAULT_MS = 3200
TOAST_LEVELS = {
    "info":    {"bg": "#FFFFFF", "fg": "#09090B", "border": "#E2E8F0", "accent": "#18181B"},
    "warning": {"bg": "#FEF3C7", "fg": "#92400E", "border": "#FBBF24", "accent": "#D97706"},
    "error":   {"bg": "#FEE2E2", "fg": "#991B1B", "border": "#FCA5A5", "accent": "#DC2626"},
    "success": {"bg": "#DCFCE7", "fg": "#166534", "border": "#86EFAC", "accent": "#16A34A"},
}

COLORS = {
    "bg":            "#F8F9FA",
    "card":          "#FFFFFF",
    "card_alt":      "#F4F4F5",
    "border":        "#E2E8F0",
    "border_strong": "#D4D4D8",
    "text":          "#09090B",
    "text_dim":      "#71717A",
    "text_faint":    "#A1A1AA",
    "primary":       "#18181B",
    "primary_hover": "#27272A",
    "primary_text":  "#FFFFFF",
    "danger":        "#DC2626",
    "success":       "#16A34A",
    "warning":       "#D97706",
    "ripple_start":  "#C4C4C8",
    "ripple_end":    "#F0F0F4",
    "glow_outer":    "#D4D4D8",
    "glow_inner":    "#E4E4E7",
    "aurora_a":      "#F1F5F9",
    "aurora_b":      "#E2E8F0",
}

# 粒子系统调色板（Google Antigravity 风格的鲜明色系，不走"夜店紫")
PARTICLE_PALETTE = [
    "#3B82F6",  # 鲜亮蓝
    "#F97316",  # 暖橘
    "#EF4444",  # 朱红
    "#06B6D4",  # 青蓝
    "#10B981",  # 翠绿
    "#F59E0B",  # 琥珀
    "#0EA5E9",  # 天蓝
    "#84CC16",  # 草绿
]
PARTICLE_NEUTRAL_COLOR = "#CBD5E1"  # 中性浅灰（少量用）

PAD = 24

# ----------------------------------------------------------------------------
# 引擎/模型候选
# ----------------------------------------------------------------------------

ENGINE_CANDIDATES_WIN = ("realesrgan-ncnn-vulkan.exe",)
ENGINE_CANDIDATES_NIX = ("realesrgan-ncnn-vulkan",)
ENGINE_CANDIDATES = ENGINE_CANDIDATES_WIN if os.name == "nt" else ENGINE_CANDIDATES_NIX

ONNX_MODEL_CANDIDATES = (
    "realesrgan-x4plus.onnx",
    "RealESRGAN_x4plus.onnx",
    "realesrgan_x4plus.onnx",
)

# 实际随包发布的 Vulkan ncnn 模型（engine/models/ 里有对应 .param/.bin 的）。
# UI 下拉框 / 智能推荐 / 配置恢复只允许出现这份名单里的名字，
# 否则引擎会报 "_wfopen ... models/xxx.param failed"。
VULKAN_SHIPPED_MODELS = (
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
)

DEFAULT_CONFIG: dict = {
    "last_opened_path": "",
    "scale_factor": "4x",
    "model_name": "realesrgan-x4plus",
}
CONFIG_FILENAME = "config.json"


# ===========================================================================
# 路径/根目录（彻底解决 C 盘硬编码）
# ===========================================================================

def get_app_root_dir() -> Path:
    """返回应用绝对根目录。
    - PyInstaller onefile 打包：sys.executable 所在目录（用户安装位置，**不是** _MEI* 临时目录）
    - 源码运行：__file__ 所在目录
    严禁返回 C:\\Windows、C:\\Users、%TEMP% 等系统目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_engine_path() -> Optional[str]:
    """在【应用根目录】内查找 Vulkan 引擎可执行文件。

    优先级：
      1. PyInstaller _MEIPASS/engine/（打包时 --add-data 注入）
      2. <root>/engine/realesrgan-ncnn-vulkan[.exe]
      3. 递归遍历 <root> 全部子目录

    严禁读取系统目录、PATH、C:\\Windows、C:\\Users。
    """
    candidates = list(ENGINE_CANDIDATES)

    # 1. PyInstaller 临时目录
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        mp = Path(meipass)
        for name in candidates:
            direct = mp / "engine" / name
            if direct.is_file():
                return str(direct)
        try:
            for name in candidates:
                for hit in mp.rglob(name):
                    if hit.is_file():
                        return str(hit)
        except OSError:
            pass

    # 2-3. 应用根目录
    root = get_app_root_dir()
    for name in candidates:
        direct = root / "engine" / name
        if direct.is_file():
            return str(direct)
    try:
        for name in candidates:
            for hit in root.rglob(name):
                if hit.is_file():
                    return str(hit)
    except OSError:
        pass
    return None


def find_vulkan_models_dir(exe_path: str) -> Optional[Path]:
    """根据 realesrgan-ncnn-vulkan.exe 路径，反推出其应使用的 models/ 目录。

    候选优先级（按顺序找，第一个含 realesrgan-x4plus.param 且 >1KB 的目录命中）：

      1. <exe_dir>/models/                ← EXE 旁路（PyInstaller onefile 临时目录）
      2. <root>/engine/models/            ← EXE/源码 视角下的 root（即应用根目录的 engine/models）
                                            — 源码运行：<源码>/engine/models
                                            — EXE  运行：<EXE所在目录>/engine/models
      3. <root>/../engine/models/         ← **关键防呆** — EXE 在 dist/ 子目录时，
                                            自动向上找源码根的 engine/models
      4. <root>/models/                   ← 兜底

    返回 None 表示完全找不到可用的模型目录。
    """
    try:
        exe_dir = Path(exe_path).resolve().parent
    except OSError:
        return None

    candidates: List[Path] = [exe_dir / "models"]
    root = get_app_root_dir()
    if root is not None:
        try:
            root_resolved = root.resolve()
        except OSError:
            root_resolved = root
        candidates.append(root / "engine" / "models")
        # 防呆：EXE 在 dist/ 子目录、模型在源码根的 engine/models
        try:
            parent = root_resolved.parent
            if parent != root_resolved:
                candidates.append(parent / "engine" / "models")
        except OSError:
            pass
        candidates.append(root / "models")

    for cand in candidates:
        if not cand.is_dir():
            continue
        probe = cand / "realesrgan-x4plus.param"
        try:
            if probe.is_file() and probe.stat().st_size > 1024:
                return cand
        except OSError:
            continue
    return None


def find_onnx_model_path() -> Optional[str]:
    """在【应用根目录】/models/ 下查找 ONNX 权重文件。"""
    models_dir = get_app_root_dir() / "models"
    if not models_dir.is_dir():
        return None
    for name in ONNX_MODEL_CANDIDATES:
        direct = models_dir / name
        if direct.is_file() and direct.stat().st_size > 1_000_000:
            return str(direct)
    return None


def get_config_path() -> Path:
    """配置文件位置。
    - 源码运行：应用根目录（方便调试）
    - PyInstaller 打包后：%LOCALAPPDATA%\\Yupianxiangsu\\config.json
      （装进 Program Files 后应用根目录不可写，写盘会静默失败）
    """
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "Yupianxiangsu" / CONFIG_FILENAME
    return get_app_root_dir() / CONFIG_FILENAME


# ===========================================================================
# 配置文件（持久化）
# ===========================================================================

def load_config() -> dict:
    """读取并解析 config.json，缺失字段用 DEFAULT_CONFIG 补齐。"""
    cfg = dict(DEFAULT_CONFIG)
    p = get_config_path()
    if not p.is_file():
        return cfg
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    if isinstance(raw, dict):
        for k, v in DEFAULT_CONFIG.items():
            cfg[k] = raw.get(k, v)
    return cfg


def save_config(cfg: dict) -> bool:
    """写入 config.json。失败返回 False（不阻塞主流程）。"""
    p = get_config_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


# ===========================================================================
# 依赖检测
# ===========================================================================

def check_msvcp140_available() -> bool:
    """检测系统是否已安装 MSVCP140.dll（VC++ 2015+ Redistributable）。"""
    if os.name != "nt":
        return True  # 非 Windows 平台无此依赖
    try:
        ctypes.WinDLL("MSVCP140.dll")
        return True
    except OSError:
        return False


# ===========================================================================
# 工具函数
# ===========================================================================

def collect_images(root: Path) -> List[Path]:
    """递归收集目录下所有合法图片文件。"""
    images: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def format_size(n: int) -> str:
    """字节数 → 人类可读尺寸。"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def default_output_for(files: List[Path]) -> str:
    """根据首张已选文件推算默认输出目录。"""
    if not files:
        return ""
    try:
        parent = files[0].resolve().parent
    except OSError:
        parent = files[0].parent
    return str(parent / "output")


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


# ===========================================================================
# ONNX CPU 推理引擎（保底模式）
# ===========================================================================

class OnnxUpscaler:
    """使用 onnxruntime 在 CPU 上做 Real-ESRGAN 超分。

    单例：进程内只加载一次 .onnx 模型（约 200~400MB），线程安全（onnxruntime
    InferenceSession 的 run() 是 GIL 安全的；并发请求会被串行执行）。
    """

    _instance: Optional["OnnxUpscaler"] = None
    _lock = threading.Lock()

    def __init__(self, model_path: str) -> None:
        if not NUMPY_OK:
            raise RuntimeError("numpy 未安装，无法启用 ONNX CPU 模式。")
        if not PIL_OK:
            raise RuntimeError("Pillow 未安装，无法启用 ONNX CPU 模式。")
        if not ONNX_RUNTIME_OK:
            raise RuntimeError("onnxruntime 未安装，无法启用 ONNX CPU 模式。")
        # 限定纯 CPU provider
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_path = model_path
        self.baked_scale = 4  # x4plus 模型固定 4 倍
        # 缓存最新一张图片的源尺寸 → 切片判断
        self._last_src_size: Tuple[int, int] = (0, 0)

    # ----- 单例工厂（线程安全） -----
    @classmethod
    def get(cls, model_path: str) -> "OnnxUpscaler":
        with cls._lock:
            if cls._instance is None or cls._instance.model_path != model_path:
                cls._instance = cls(model_path)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    # ----- 单图推理 -----
    # 直接整图推理的长边上限：超过就切 tile（RRDB 全图跑大图会吃几十 GB 内存）
    DIRECT_LIMIT_PX = 384
    ONNX_TILE_PX = 256    # 输入空间 tile 边长
    ONNX_OVERLAP_PX = 16  # 输入空间重叠

    def _out_to_image(self, out: "np.ndarray") -> "Image.Image":
        """ONNX 原始输出 → PIL 图像（含 shape 校验）。"""
        if out.ndim != 4 or out.shape[0] != 1 or out.shape[1] != 3:
            raise RuntimeError(f"ONNX 输出 shape 异常：{out.shape}")
        out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)  # HWC
        return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))

    def _infer_full(self, img: "Image.Image") -> "Image.Image":
        """整图推理（仅小图）。"""
        arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC [0,1]
        arr = arr.transpose(2, 0, 1)[None, ...]           # NCHW
        outputs = self.session.run([self.output_name], {self.input_name: arr})
        return self._out_to_image(outputs[0])

    def _infer_tiled(self, img: "Image.Image") -> "Image.Image":
        """切片推理 + 无缝拼接。

        拼接规则（与 _tiled_vulkan_upscale 一致）：
          - step = tile - overlap（输入空间）
          - 非 0 列/行只裁左侧/上侧 overlap×scale 的输出重叠
          - 右/下边缘不裁 → 无缝无空洞
        """
        w, h = img.size
        s = self.baked_scale
        tile, ov = self.ONNX_TILE_PX, self.ONNX_OVERLAP_PX
        step = tile - ov
        canvas = Image.new("RGB", (w * s, h * s))
        y0 = 0
        while y0 < h:
            x0 = 0
            y1 = min(h, y0 + tile)
            while x0 < w:
                x1 = min(w, x0 + tile)
                up = self._infer_full(img.crop((x0, y0, x1, y1)))
                uw, uh = up.size
                lx0 = ov * s if x0 > 0 else 0
                ly0 = ov * s if y0 > 0 else 0
                if uw > lx0 and uh > ly0:
                    canvas.paste(
                        up.crop((lx0, ly0, uw, uh)),
                        (x0 * s + lx0, y0 * s + ly0),
                    )
                if x1 >= w:
                    break
                x0 += step
            if y1 >= h:
                break
            y0 += step
        return canvas

    def run(self, src: Path, dst: Path) -> Tuple[bool, str]:
        """读取 → 推理（小图整跑 / 大图切片）→ 保存。返回 (成功, 错误详情)。"""
        try:
            img = Image.open(src).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return False, f"读取图片失败：{exc}"

        w, h = img.size
        self._last_src_size = (w, h)
        # x4plus 模型固定 4x 输出
        out_w, out_h = w * self.baked_scale, h * self.baked_scale
        try:
            if max(w, h) <= self.DIRECT_LIMIT_PX:
                out_img = self._infer_full(img)
            else:
                out_img = self._infer_tiled(img)
        except Exception as exc:  # noqa: BLE001
            return False, f"ONNX 推理失败：{exc}"
        # 兜底：模型输出若与期望尺寸不符，按用户选择缩放
        if out_img.size != (out_w, out_h):
            out_img = out_img.resize((out_w, out_h), Image.LANCZOS)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # 按目标后缀选格式：.jpg 用 JPEG(95)，其它一律 PNG
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                out_img.save(dst, format="JPEG", quality=95, optimize=True)
            else:
                out_img.save(dst, format="PNG")
        except Exception as exc:  # noqa: BLE001
            return False, f"保存图片失败：{exc}"
        return True, ""


# ===========================================================================
# Vulkan 子进程封装 + GPU/CPU 自动降级（Vulkan 引擎内部）
# ===========================================================================

def _run_subprocess(cmd: list[str], cwd: Optional[str]) -> Tuple[bool, str]:
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            # 1800s：4K 级联（两遍 Vulkan）+ 小 tile + 单 job 在慢 GPU 上
            # 可能远超 10 分钟，杀早了 = 白跑
            timeout=1800, creationflags=flags, cwd=cwd,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            return False, err or f"退出码 {result.returncode}"
        return True, ""
    except FileNotFoundError:
        return False, "找不到 realesrgan-ncnn-vulkan 可执行文件"
    except subprocess.TimeoutExpired:
        return False, "处理超时（超过 30 分钟）"
    except OSError as exc:
        return False, f"无法启动外部引擎：{exc}"


def _is_hardware_error(err: str) -> bool:
    """判断 Vulkan 子进程错误是否属于硬件层失败（需要降级）。"""
    if not err:
        return False
    k = err.lower()
    return any(s in k for s in (
        "vulkan", "device", "failed to create", "no vulkan", "no device",
        "gpu", "out of memory", "no available", "instance", "driver",
    ))


def _upscale_vulkan(
    exe_path: str, src: Path, dst: Path, scale: int, model: str
) -> Tuple[bool, str, bool]:
    """调用 Vulkan 引擎。返回 (ok, err, hardware_failed)。

    关键：通过 ``-m`` 显式指定 models 目录，避免依赖 EXE 的 cwd 推断。
    PyInstaller onefile 模式下，EXE 内部的 models/ 经常是空的（精简包只
    含 EXE/DLL），所以必须让引擎走【用户安装位置】的 engine/models/。
    """
    cwd = str(Path(exe_path).resolve().parent)
    models_dir = find_vulkan_models_dir(exe_path)
    base_cmd = [exe_path, "-i", str(src), "-o", str(dst),
                "-s", str(scale), "-n", model]
    if models_dir is not None:
        base_cmd += ["-m", str(models_dir)]

    # -j 1:1:1 = 单 GPU job（避免双倍显存）
    # -t 128   = 小 tile（避免中间 buffer 吃光显存）
    gpu_cmd = base_cmd + ["-j", "1:1:1", "-t", str(VULKAN_TILE_SIZE), "-g", "0"]
    ok, err = _run_subprocess(gpu_cmd, cwd)
    if ok:
        if dst.is_file():
            return True, "", False
        return False, "命令执行成功但未生成超分图片", False
    if not _is_hardware_error(err):
        return False, err, False

    # Vulkan 引擎的 -g -1 CPU 模式（实测 20220424 版引擎会直接报
    # "invalid gpu device"，此路径大概率失败；真正的 CPU 兜底在上层 ONNX）
    cpu_cmd = base_cmd + ["-g", "-1", "-t", "100"]
    ok2, err2 = _run_subprocess(cpu_cmd, cwd)
    if ok2 and dst.is_file():
        return True, "", True  # 成功，但用 CPU 跑的
    return False, err2 or err, True


def _upscale_onnx(
    model_path: str, src: Path, dst: Path, scale: int
) -> Tuple[bool, str]:
    """调用 ONNX CPU 引擎。注意：x4plus 模型固定 4 倍。"""
    if scale != 4:
        return False, f"ONNX CPU 引擎当前仅支持 4x（所选 {scale}x 需对应 ONNX 模型）"
    try:
        engine = OnnxUpscaler.get(model_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"加载 ONNX 模型失败：{exc}"
    return engine.run(src, dst)


# ===========================================================================
# 目标档位（4K/2K/FHD）专用流水线
# ----------------------------------------------------------------------------
# 核心思想：1 次 Vulkan x4 + LANCZOS 补尺寸 + USM/CLAHE 软化
# - 模糊度预检（<1ms）：原图清晰 → 跳过 Vulkan，**最省 GPU**
# - Vulkan AI 只跑 1 次（避免"对 AI 输出再 AI"产生的伪影正反馈）
# - 尺寸补到目标长边用 LANCZOS（CPU 毫秒级，数学保形）
# - USM(r=1.2, amt=0.9, thr=5) + σ=0.35 高斯软化 + CLAHE(clip=1.8, 8x8)
# - 三层 fallback：Vulkan → ONNX CPU → 纯 LANCZOS（保证有输出）
# ===========================================================================

def _laplacian_blur_score(img: "Image.Image") -> float:
    """模糊度快速判读：Laplacian 方差。值越大越清晰。
    用 PIL ImageFilter.Kernel 直接走 C 实现，< 1ms。
    """
    try:
        g = img.convert("L")
    except Exception:  # noqa: BLE001
        return 0.0
    w, h = g.size
    # 缩到 ≤ 256 长边再算：精度对"清/糊"判读足够，速度快 5-10x
    if max(w, h) > 256:
        scale = 256.0 / max(w, h)
        g = g.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                     Image.Resampling.LANCZOS)
    try:
        # 3x3 Laplacian kernel
        edge = g.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1.0, offset=128,
        ))
        # 方差 = 清晰度代理
        stat = edge.getdata()
        # 1-pass variance via mean
        s = 0
        s2 = 0
        n = 0
        for v in stat:
            s += v
            s2 += v * v
            n += 1
        if n == 0:
            return 0.0
        mean = s / n
        return max(0.0, s2 / n - mean * mean)
    except Exception:  # noqa: BLE001
        return 0.0


def _resize_to_long_edge(img: "Image.Image", target_long: int) -> "Image.Image":
    """按比例缩放到 target_long 长边。放大缩小都做（永不裁切、永不变形）。"""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge == target_long:
        return img
    ratio = target_long / float(long_edge)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _postprocess_linemelt(img: "Image.Image") -> "Image.Image":
    """线条软化：USM(温和版) + YCbCr CLAHE。

    设计原则：
    - USM 只动大尺度边缘（threshold=10），避开模型已经绘制的细线/纹理
    - **不再叠加高斯模糊** — 高斯会无差别糊化所有细节，是糊的元凶
    - CLAHE 在 YCbCr 空间只动 Y 亮度（HSV 会动 V 让色彩偏移）
    - clip_limit 克制到 1.4，宁可不过曝
    """
    if not PIL_OK:
        return img
    # 1) USM 温和版：threshold=10（提一档，避免在小边缘上画 halo）
    #    percent=60（降一档，避免锐过头）
    try:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=10))
    except Exception:  # noqa: BLE001
        pass
    # 2) CLAHE YCbCr 版：只动 Y 亮度，色彩不动
    if NUMPY_OK:
        try:
            ycbcr = img.convert("YCbCr")
            arr = np.asarray(ycbcr, dtype=np.uint8)
            y = arr[..., 0]
            h, w = y.shape
            tile = 8
            th = max(1, h // tile)
            tw = max(1, w // tile)
            small_h = max(1, h // 4)
            small_w = max(1, w // 4)
            y_small = np.asarray(
                Image.fromarray(y).resize((small_w, small_h),
                                          Image.Resampling.BILINEAR)
            )
            sh, sw = y_small.shape
            sth = max(1, sh // tile)
            stw = max(1, sw // tile)
            clip_lim = 1.4
            out = np.empty_like(y_small)
            for ty in range(tile):
                for tx in range(tile):
                    y0 = ty * sth
                    x0 = tx * stw
                    y1 = sh if ty == tile - 1 else (ty + 1) * sth
                    x1 = sw if tx == tile - 1 else (tx + 1) * stw
                    block = y_small[y0:y1, x0:x1]
                    bh, bw = block.shape
                    clip_int = max(1, int(clip_lim * bh * bw / 256.0))
                    hist, _ = np.histogram(block, bins=256, range=(0, 256))
                    excess = np.where(hist > clip_int, hist - clip_int, 0)
                    hist = np.minimum(hist, clip_int).astype(np.float64)
                    if excess.sum() > 0 and hist.sum() > 0:
                        hist = hist + excess.sum() * (hist / hist.sum())
                    cdf = np.cumsum(hist)
                    if cdf[-1] == 0:
                        out[y0:y1, x0:x1] = block
                        continue
                    cdf = (cdf - cdf[0]) * 255.0 / max(1e-9, cdf[-1] - cdf[0])
                    cdf = np.clip(cdf, 0, 255).astype(np.uint8)
                    out[y0:y1, x0:x1] = cdf[block]
            eq = np.asarray(
                Image.fromarray(out).resize((w, h), Image.Resampling.BILINEAR)
            )
            mixed = (y.astype(np.float32) * 0.7
                     + eq.astype(np.float32) * 0.3).clip(0, 255).astype(np.uint8)
            arr[..., 0] = mixed
            img = Image.fromarray(arr, mode="YCbCr").convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    return img


def _vulkan_upscale_to_intermediate(
    engine_path: str,
    src: Path,
    intermediate: Path,
    model: str,
    tile_size: Optional[int] = None,
) -> Tuple[bool, str]:
    """Vulkan AI 超分：固定 x4，输出 intermediate。
    -j 1:1:1 / -t VULKAN_TILE_SIZE = 单 job + 小 tile，显存峰值 ≤ 1.5GB
    """
    models_dir = find_vulkan_models_dir(engine_path)
    t = tile_size if tile_size is not None else VULKAN_TILE_SIZE
    base_cmd = [engine_path, "-i", str(src), "-o", str(intermediate),
                "-s", "4", "-n", model, "-j", "1:1:1", "-t", str(t), "-g", "0"]
    if models_dir is not None:
        base_cmd += ["-m", str(models_dir)]
    cwd = str(Path(engine_path).resolve().parent)
    ok, err = _run_subprocess(base_cmd, cwd)
    if ok and intermediate.is_file():
        return True, ""
    return False, err or "Vulkan 推理失败"


def _tiled_vulkan_upscale(
    engine_path: str,
    src: Path,
    intermediate: Path,
    model: str,
    tile_px: int = 512,
    overlap: int = 32,
) -> Tuple[bool, str]:
    """Vulkan 切 tile：4K 原图跑 16K 时保护显存 ≤ 800MB。

    拼接规则（输入空间语义）：
      - tile_px / overlap 均为【原图像素】
      - 步进 step = tile_px - overlap（相邻 tile 在输入空间重叠 overlap 像素）
      - 非 0 列/行只裁掉【左侧/上侧】overlap×4 的输出重叠（前一个 tile 已覆盖）
      - 右/下边缘不裁 → 保证无缝且无空洞
    """
    try:
        src_img = Image.open(src)
    except Exception as exc:  # noqa: BLE001
        return False, f"读图失败：{exc}"
    # 统一到 RGB/RGBA，避免调色板(P)等模式拼接错乱
    if src_img.mode not in ("RGB", "RGBA"):
        src_img = src_img.convert("RGBA")
    sw, sh = src_img.size
    dw, dh = sw * 4, sh * 4
    canvas = Image.new(src_img.mode, (dw, dh))
    step = tile_px - overlap          # 输入空间步进
    pad_out = overlap * 4             # overlap 在输出（x4）空间的尺寸
    y0 = 0
    while y0 < sh:
        x0 = 0
        y1 = min(sh, y0 + tile_px)
        while x0 < sw:
            x1 = min(sw, x0 + tile_px)
            crop = src_img.crop((x0, y0, x1, y1))
            tile_src = intermediate.with_suffix(f".tile_{x0}_{y0}.png")
            tile_dst = intermediate.with_suffix(f".tile_{x0}_{y0}_x4.png")
            try:
                crop.save(tile_src)
                ok, err = _vulkan_upscale_to_intermediate(engine_path, tile_src, tile_dst, model)
                if not ok:
                    return False, err
                up = Image.open(tile_dst)
                uw, uh = up.size
                # 非 0 列/行裁掉左侧/上侧 overlap（×4 输出），右侧/下侧保留
                lx0 = pad_out if x0 > 0 else 0
                ly0 = pad_out if y0 > 0 else 0
                if uw > lx0 and uh > ly0:
                    up_crop = up.crop((lx0, ly0, uw, uh))
                    canvas.paste(up_crop, (x0 * 4 + lx0, y0 * 4 + ly0))
                # uw <= lx0 说明末尾 tile 宽度不足 overlap，前一个 tile 已覆盖 → 跳过
            finally:
                tile_src.unlink(missing_ok=True)
                tile_dst.unlink(missing_ok=True)
            if x1 >= sw:
                break
            x0 += step
        if y1 >= sh:
            break
        y0 += step
    try:
        canvas.save(intermediate)
    except Exception as exc:  # noqa: BLE001
        return False, f"保存拼接结果失败：{exc}"
    return True, ""


def _target_preset_dispatch(
    engine_path: Optional[str],
    onnx_model_path: Optional[str],
    src: Path,
    dst: Path,
    target_label: str,
    model: str,
) -> Tuple[bool, str, str]:
    """4K/2K/FHD 档位的总入口。

    返回 (ok, error, engine_used):
      engine_used ∈ {"vulkan-gpu", "vulkan-tile", "vulkan-cpu",
                     "onnx-cpu", "lanczos-only", ""}
    """
    target_long = TARGET_PRESETS.get(target_label)
    if target_long is None:
        return False, f"未知的目标档位：{target_label}", ""

    try:
        src_img = Image.open(src)
    except Exception as exc:  # noqa: BLE001
        return False, f"读图失败：{exc}", ""

    long_edge = max(src_img.size)
    blur_score = _laplacian_blur_score(src_img)

    # ---- 阶段 A：决定要不要跑 Vulkan ----
    # 1) 已经是小图（长边 ≤ 目标长边）→ 直接后处理到目标
    if long_edge >= target_long:
        run_vulkan = False
        skip_reason = f"原图长边 {long_edge}px 已 ≥ 目标 {target_long}px，无需 AI 放大"
    # 2) 总放大倍数 < 4（即用 1 次 x4 + 剩余 LANCZOS 才到目标）→ Vulkan
    elif long_edge * 4 < target_long:
        run_vulkan = True
        skip_reason = ""
    # 3) 总放大倍数 ≥ 4 但 ≤ 8：x4 之后 LANCZOS 比例小，可能完全省 Vulkan
    else:
        # 比如 800x600 想跑 4K，800*4 = 3200 < 3840，需要再 LANCZOS 一点点
        # 比如 1100x800 想跑 4K，1100*4 = 4400 > 3840，要先 LANCZOS 缩到 ~960 再 Vulkan
        # 后者其实不如直接 LANCZOS → 4K，模糊度足够时（>50）不跑 Vulkan
        if blur_score > 50.0 and long_edge >= target_long / 4:
            run_vulkan = False
            skip_reason = (f"原图清晰（Laplacian={blur_score:.1f}）且放大比例适中，"
                           f"跳过 Vulkan GPU 推理，节省 GPU 资源")
        else:
            run_vulkan = True
            skip_reason = ""

    # ---- 阶段 B：执行 ----
    if not run_vulkan:
        # 直接 LANCZOS 到目标 + 后处理
        try:
            big = _resize_to_long_edge(src_img, target_long)
            big = _postprocess_linemelt(big)
            # 保存为 PNG（无损，可换 JPEG）
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                big = big.convert("RGB")
                big.save(dst, "JPEG", quality=95, optimize=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                big.save(dst)
            return True, skip_reason, "lanczos-only"
        except Exception as exc:  # noqa: BLE001
            return False, f"LANCZOS 后处理失败：{exc}", ""

    # ---- 要跑 Vulkan ----
    intermediate = dst.with_suffix(".intermediate.png")
    used_engine = "vulkan-gpu"
    if engine_path:
        # ===== 级联 Vulkan =====
        # 旧方案: 247×300 → Vulkan x4 → 988×1200 → LANCZOS 拉到 4K (3.2x 拉大, 无新细节)
        # 新方案: 247×300 → Vulkan x4 → 988×1200 → Vulkan x4 → 3952×4800 → LANCZOS 到 4K
        # 第二遍 Vulkan 处理的是 ~1.2MP 中间图, 显存安全(<1.5GB), 真补出 16x 细节
        # 决策: 当 long_edge * 4 < target_long (需要 ≥ 2 次 4x 才能到) → 级联
        needs_cascade = (long_edge * 4 < target_long) and (long_edge * 16 >= target_long / 2)
        # 万一需要 3 次 (如 100x100 → 4K), 也只级联 2 次避免伪影正反馈
        inter1 = intermediate
        inter2 = dst.with_suffix(".intermediate2.png")
        if needs_cascade:
            # 第一遍: 原图 → inter1
            ok1, err1 = _vulkan_upscale_to_intermediate(engine_path, src, inter1, model)
            if not ok1:
                ok, err = False, f"Vulkan 级联第一遍失败：{err1}"
            else:
                # 第二遍: inter1 → inter2
                ok2, err2 = _vulkan_upscale_to_intermediate(engine_path, inter1, inter2, model)
                if not ok2:
                    # 第二遍失败 → 退回到单遍（用第一遍结果继续）
                    ok, err = True, f"Vulkan 级联第二遍失败，降到单遍：{err2}"
                    used_engine = "vulkan-gpu"
                    inter2.unlink(missing_ok=True)
                else:
                    ok, err = True, ""
                    used_engine = "vulkan-cascade"
                    inter1.unlink(missing_ok=True)
                    intermediate = inter2
        else:
            # 单遍（输入已经够大或目标较小）
            if src_img.size[0] * src_img.size[1] > VULKAN_TILE_PIXEL_LIMIT:
                ok, err = _tiled_vulkan_upscale(engine_path, src, intermediate, model)
                used_engine = "vulkan-tile"
            else:
                ok, err = _vulkan_upscale_to_intermediate(engine_path, src, intermediate, model)
        if not ok:
            # Vulkan 失败 → ONNX 兜底
            if onnx_model_path and ONNX_RUNTIME_OK and PIL_OK and NUMPY_OK:
                try:
                    engine = OnnxUpscaler.get(onnx_model_path)
                    ok2, err2 = engine.run(src, intermediate)
                    if ok2 and intermediate.is_file():
                        ok, err = True, ""
                        used_engine = "onnx-cpu"
                    else:
                        err = f"Vulkan 失败：{err}; ONNX 兜底失败：{err2}"
                except Exception as exc:  # noqa: BLE001
                    err = f"Vulkan 失败：{err}; ONNX 异常：{exc}"
            if not ok:
                # 兜底：纯 LANCZOS 到目标（保证有输出）
                try:
                    big = _resize_to_long_edge(src_img, target_long)
                    big = _postprocess_linemelt(big)
                    if dst.suffix.lower() in (".jpg", ".jpeg"):
                        big = big.convert("RGB")
                        big.save(dst, "JPEG", quality=95, optimize=True)
                    else:
                        big.save(dst)
                    return True, f"Vulkan + ONNX 均失败，LANCZOS 兜底：{err}", "lanczos-only"
                except Exception as exc2:  # noqa: BLE001
                    return False, f"全链路失败：Vulkan={err}; LANCZOS={exc2}", ""

    elif onnx_model_path and ONNX_RUNTIME_OK and PIL_OK and NUMPY_OK:
        try:
            engine = OnnxUpscaler.get(onnx_model_path)
            ok, err = engine.run(src, intermediate)
            used_engine = "onnx-cpu"
        except Exception as exc:  # noqa: BLE001
            ok, err = False, str(exc)
        if not ok:
            try:
                big = _resize_to_long_edge(src_img, target_long)
                big = _postprocess_linemelt(big)
                if dst.suffix.lower() in (".jpg", ".jpeg"):
                    big = big.convert("RGB")
                    big.save(dst, "JPEG", quality=95, optimize=True)
                else:
                    big.save(dst)
                return True, f"ONNX 失败，LANCZOS 兜底：{err}", "lanczos-only"
            except Exception as exc2:  # noqa: BLE001
                return False, f"全链路失败：ONNX={err}; LANCZOS={exc2}", ""
    else:
        # 没有任何 AI 引擎 → 纯 LANCZOS
        try:
            big = _resize_to_long_edge(src_img, target_long)
            big = _postprocess_linemelt(big)
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                big = big.convert("RGB")
                big.save(dst, "JPEG", quality=95, optimize=True)
            else:
                big.save(dst)
            return True, "无 AI 引擎可用，已用 LANCZOS + 软化处理", "lanczos-only"
        except Exception as exc:  # noqa: BLE001
            return False, f"LANCZOS 失败：{exc}", ""

    # ---- 阶段 C：LANCZOS 补到目标长边 + 后处理 ----
    try:
        up = Image.open(intermediate)
        big = _resize_to_long_edge(up, target_long)
        big = _postprocess_linemelt(big)
        if dst.suffix.lower() in (".jpg", ".jpeg"):
            big = big.convert("RGB")
            big.save(dst, "JPEG", quality=95, optimize=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            big.save(dst)
        return True, "", used_engine
    except Exception as exc:  # noqa: BLE001
        return False, f"后处理失败：{exc}", used_engine
    finally:
        intermediate.unlink(missing_ok=True)


# ===========================================================================
# 引擎调度：Vulkan 优先 → ONNX 兜底（彻底保底）
# ===========================================================================

def _dispatch_upscaler(
    engine_path: Optional[str],
    onnx_model_path: Optional[str],
    src: Path,
    dst: Path,
    scale: int,
    model: str,
) -> Tuple[bool, str, str]:
    """引擎调度主函数。

    返回 (ok, error, engine_used)：
      engine_used ∈ {"vulkan-gpu", "vulkan-cpu", "onnx-cpu", ""}
    """
    # 1) Vulkan 优先
    if engine_path:
        ok, err, hw_fail = _upscale_vulkan(engine_path, src, dst, scale, model)
        if ok:
            return True, "", ("vulkan-cpu" if hw_fail else "vulkan-gpu")
        # 硬件层失败：尝试 ONNX 兜底
        if hw_fail and onnx_model_path and scale == 4 and ONNX_RUNTIME_OK and PIL_OK and NUMPY_OK:
            ok2, err2 = _upscale_onnx(onnx_model_path, src, dst, scale)
            if ok2:
                return True, "", "onnx-cpu"
            return False, f"Vulkan 硬件不可用，ONNX 兜底也失败：{err2}", ""
        return False, err, ""

    # 2) 没有任何 Vulkan 引擎 → 直接走 ONNX
    if onnx_model_path and scale == 4 and ONNX_RUNTIME_OK and PIL_OK and NUMPY_OK:
        ok, err = _upscale_onnx(onnx_model_path, src, dst, scale)
        if ok:
            return True, "", "onnx-cpu"
        return False, err, ""

    return False, "无可用引擎（Vulkan 未找到且 ONNX 模型/依赖缺失）", ""


# ===========================================================================
# 后台 Worker
# ===========================================================================

class UpscaleWorker:
    """后台线程批量放大图片（双引擎智能调度）。"""

    def __init__(
        self,
        engine_path: Optional[str],
        onnx_model_path: Optional[str],
        files: List[Path],
        output_dir: Path,
        scale: int,
        model: str,
        log: Callable[[str, str], None],
        on_finished: Callable[[bool], None],
        stop_event: threading.Event,
        max_workers: int = MAX_WORKERS,
        on_progress: Optional[Callable[[int, int], None]] = None,
        target_preset: str = "",
    ) -> None:
        self.engine_path = engine_path
        self.onnx_model_path = onnx_model_path
        self.files = files
        self.output_dir = output_dir
        self.scale = scale
        self.model = model
        self.log = log
        self.on_finished = on_finished
        self.stop_event = stop_event
        self.max_workers = max(1, max_workers)
        self.on_progress = on_progress
        # 4K/2K/FHD 档位（为空时走原 scale 路径）
        self.target_preset = target_preset
        # 记录本次实际使用的引擎（用于日志/气泡）
        self._engine_used: set = set()

    def _report(self, done: int, total: int) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(done, total)
        except Exception:  # noqa: BLE001
            pass

    def _process_one(self, src: Path, dst: Path) -> Tuple[bool, str, str]:
        if self.target_preset and self.target_preset in TARGET_PRESETS:
            return _target_preset_dispatch(
                self.engine_path, self.onnx_model_path,
                src, dst, self.target_preset, self.model,
            )
        return _dispatch_upscaler(
            self.engine_path, self.onnx_model_path,
            src, dst, self.scale, self.model,
        )

    def run(self) -> None:
        success_count = 0
        fail_count = 0
        try:
            total = len(self.files)
            if total == 0:
                self.log("未选择任何图片文件。", "warning")
                self.on_finished(False)
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._engine_used.clear()
            # 启动横幅：哪个引擎扛活
            engines = []
            if self.engine_path:
                engines.append("Vulkan")
            if self.onnx_model_path and (self.scale == 4 or self.target_preset):
                engines.append("ONNX")
            scale_label = f"{self.target_preset}档" if self.target_preset else f"{self.scale}x"
            self.log(
                f"开始处理 {total} 张图片（{scale_label}，并发={self.max_workers}，"
                f"可用引擎: {' / '.join(engines) or '无'}）…",
                "info",
            )
            if self.engine_path:
                self.log("[硬件加速] 已成功启用 Vulkan GPU 硬件加速引擎", "success")
            if self.onnx_model_path and (self.scale == 4 or self.target_preset):
                self.log(
                    "[保底模式] 已就绪 ONNX CPU 引擎（无 GPU 时自动降级使用）",
                    "info",
                )

            # 规划输出路径（去重 + 倍率后缀）
            plan: List[Tuple[Path, Path]] = []
            used: set = set()
            suffix = self.target_preset.lower() if self.target_preset else f"{self.scale}x"
            for src in self.files:
                if self.stop_event.is_set():
                    break
                stem, suf = src.stem, src.suffix
                out_name = f"{stem}_{suffix}{suf}"
                if out_name in used:
                    n = 2
                    while f"{stem}_{suffix}_{n}{suf}" in used:
                        n += 1
                    out_name = f"{stem}_{suffix}_{n}{suf}"
                used.add(out_name)
                plan.append((src, self.output_dir / out_name))

            if not plan:
                self.log("用户已取消处理。", "warning")
                self.on_finished(False)
                return

            # 线程池并发执行
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_map = {
                    pool.submit(self._process_one, s, d): (idx, s, d)
                    for idx, (s, d) in enumerate(plan, start=1)
                }
                for future in as_completed(future_map):
                    if self.stop_event.is_set():
                        self.log("用户已取消处理。", "warning")
                        break
                    idx, src, dst = future_map[future]
                    try:
                        ok, detail, engine_used = future.result()
                    except Exception as exc:  # noqa: BLE001
                        ok, detail, engine_used = False, f"线程异常：{exc}", ""
                    if engine_used:
                        self._engine_used.add(engine_used)
                    if ok:
                        success_count += 1
                        # detail 非 0 = 有值得告诉用户的信息（如"跳过 Vulkan 省资源"）
                        note = engine_used
                        if detail:
                            note = f"{note} · {detail}" if note else detail
                        self.log(
                            f"[{idx}/{total}] ✓ {src.name} → {dst.name}  ({note})",
                            "success",
                        )
                    else:
                        fail_count += 1
                        self.log(
                            f"[{idx}/{total}] ✗ {src.name}：{detail}", "error"
                        )
                    self._report(success_count + fail_count, total)

            # 引擎实际使用统计
            if self._engine_used:
                self.log(
                    "本次实际使用引擎: " + " + ".join(sorted(self._engine_used)),
                    "info",
                )
            self.log(
                f"处理结束：成功 {success_count}，失败 {fail_count}，共 {total}。",
                "info" if fail_count == 0 and not self.stop_event.is_set() else "warning",
            )
            self.on_finished(fail_count == 0 and not self.stop_event.is_set())
        except Exception as exc:  # noqa: BLE001
            self.log(f"处理过程出现异常：{exc}", "error")
            self.on_finished(False)


# ===========================================================================
# Toast 气泡（顶部非模态提示，不阻塞 UI）
# ===========================================================================

class Toast(tk.Toplevel):
    """轻量气泡：顶部居中滑入，3 秒后自动消失。"""

    def __init__(
        self, parent: tk.Misc, message: str, level: str = "info",
        duration_ms: int = TOAST_DEFAULT_MS,
    ) -> None:
        super().__init__(parent)
        self._duration_ms = duration_ms
        self._level = level

        # 无边框 + 置顶
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        try:
            # Windows 半透明（其它平台可能抛 TclError）
            self.attributes("-alpha", 0.97)
        except tk.TclError:
            pass

        style = TOAST_LEVELS.get(level, TOAST_LEVELS["info"])
        # 用 Frame 画左色条 + 文字
        outer = tk.Frame(self, bg=style["bg"], highlightbackground=style["border"],
                         highlightthickness=1, bd=0)
        outer.pack(fill="both", expand=True)
        bar = tk.Frame(outer, bg=style["accent"], width=4)
        bar.pack(side="left", fill="y")
        text_frame = tk.Frame(outer, bg=style["bg"])
        text_frame.pack(side="left", fill="both", expand=True, padx=12, pady=10)
        lbl = tk.Label(
            text_frame, text=message,
            bg=style["bg"], fg=style["fg"],
            font=("Microsoft YaHei UI", 10, "bold"),
            justify="left", anchor="w", wraplength=420,
        )
        lbl.pack(side="left", fill="x", expand=True)
        # 点击关闭
        for w in (self, outer, text_frame, lbl, bar):
            w.bind("<Button-1>", lambda e: self._dismiss())
        # 自动布局 + 定位
        self.update_idletasks()
        self._position(parent)
        # 淡入
        self._fade_in()
        # 自动消失
        self.after(duration_ms, self._dismiss)

    def _position(self, parent: tk.Misc) -> None:
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            w = self.winfo_width()
            h = self.winfo_height()
            x = px + (pw - w) // 2
            y = py + 16
            self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except tk.TclError:
            pass

    def _fade_in(self) -> None:
        try:
            for a in (0.4, 0.6, 0.8, 0.97):
                self.attributes("-alpha", a)
                self.update_idletasks()
        except tk.TclError:
            pass

    def _dismiss(self) -> None:
        try:
            self.destroy()
        except tk.TclError:
            pass


# ===========================================================================
# 粒子背景：Google Antigravity 风格（直接照抄 HTML data-* 参数）
# ===========================================================================
#
# 参考 HTML（用户提供的真·反重力源码）：
#   <div class="main-particles-component-section"
#        data-theme="light"
#        data-ring-width="0.006"
#        data-ring-width2="0.107"
#        data-ring-displacement="0.62"
#        data-density="230"
#        data-particles-scale="0.59">
# 我们的 tkinter 实现保留这些参数的语义：
#   - density=230       → 230 个粒子（够密）
#   - particles_scale   → 粒子半径（0.59 ≈ 1.5px）
#   - ring_displacement → 鼠标对环的位移强度（0.62，强）
#   - theme=light       → 极简白配色
# ===========================================================================

class AntigravityParticles:
    """Google Antigravity 同心环粒子系统（tkinter 实现版）。

    视觉：
      - 多层同心环，每环不同半径 / 粒子数 / 旋转速度
      - 粒子沿环均匀分布 + 角度微抖，看起来像"液体环"
      - 鼠标在画布内形成径向位移场，把附近粒子推开 → 环被"压扁"
      - 外环用鲜明彩色（蓝/橘/红/青/绿/琥珀），内环用浅灰

    性能：230 粒子 + 33ms tick + 30fps 在普通笔记本稳定运行。
    """

    def __init__(
        self,
        canvas: "tk.Canvas",
        *,
        theme: str = "light",
        density: int = 230,
        ring_count: int = 5,
        particles_scale: float = 0.59,
        ring_displacement: float = 0.62,
        mouse_radius: int = 180,
    ) -> None:
        self.canvas = canvas
        self.theme = theme
        self.density = density
        self.ring_count = ring_count
        self.particles_scale = particles_scale
        self.ring_displacement = ring_displacement
        self.mouse_radius = mouse_radius
        self.mouse_force = ring_displacement  # 0.62，Google 原值
        self.mouse_x = -9999.0
        self.mouse_y = -9999.0
        self.particles: List[dict] = []
        self._dot_items: List[int] = []
        self._tick_handle: Optional[str] = None
        self._seed = 1337
        # 配色
        if theme == "light":
            self.neutral_color = "#94A3B8"  # 浅灰
            # 暖 + 冷色：跟 Google 一致（蓝/橘/红/青/绿/琥珀）
            self.accent_palette = [
                "#3B82F6", "#F97316", "#EF4444",
                "#06B6D4", "#10B981", "#F59E0B",
            ]
        else:  # dark
            self.neutral_color = "#E2E8F0"
            self.accent_palette = [
                "#FF4641", "#346BF1", "#FFB800", "#10B981",
            ]
        self._init()

    @property
    def count(self) -> int:
        return len(self.particles)

    def _rand(self) -> float:
        self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF
        return self._seed / 0x7FFFFFFF

    def _init(self) -> None:
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        if w < 100 or h < 100:
            return
        # 清旧
        for it in self._dot_items:
            try:
                self.canvas.delete(it)
            except tk.TclError:
                pass
        self.canvas.delete("dot", "line")
        self._dot_items = []
        self.particles = []
        self.cw, self.ch = float(w), float(h)
        self.cx, self.cy = w * 0.5, h * 0.5
        # 最大半径（用屏幕对角线一半的 70% — Google 那种环绕感）
        self.max_r = min(w, h) * 0.7
        self.min_r = min(w, h) * 0.04
        # 同心环结构：5 环（参数 ring_count）
        # 每环的密度按 (1 + 2*r) 分配（外环粒子更多）
        ring_weights = [1 + 2.0 * (i / max(1, self.ring_count - 1)) for i in range(self.ring_count)]
        total_w = sum(ring_weights)
        ring_counts = [max(8, int(self.density * w_ / total_w)) for w_ in ring_weights]
        # 调整到总数对齐
        diff = self.density - sum(ring_counts)
        if diff > 0:
            ring_counts[-1] += diff
        elif diff < 0:
            for i in range(len(ring_counts)):
                cut = min(ring_counts[i] - 8, -diff)
                if cut > 0:
                    ring_counts[i] -= cut
                    diff += cut
                    if diff == 0:
                        break
        # 旋转速度：内环快、外环慢（Google 那种内层更"转动"的感觉）
        ring_speeds = [
            (1.0 - 0.7 * (i / max(1, self.ring_count - 1))) * 0.0004
            * (1 if i % 2 == 0 else -1)  # 奇偶环方向相反
            for i in range(self.ring_count)
        ]
        # 创建粒子
        idx = 0
        for ring in range(self.ring_count):
            r_norm = ring / max(1, self.ring_count - 1)  # 0..1
            ring_r = self.min_r + r_norm * (self.max_r - self.min_r)
            n = ring_counts[ring]
            angle_step = 2 * math.pi / n
            speed = ring_speeds[ring]
            for i in range(n):
                if idx >= self.density:
                    break
                # 基础角度 + 抖动（避免完全等距）
                base_angle = i * angle_step + (self._rand() - 0.5) * angle_step * 0.5
                # 半径抖动（环有厚度，不是数学圆）
                r_jitter = (self._rand() - 0.5) * (self.max_r - self.min_r) * 0.04
                r = ring_r + r_jitter
                # 颜色：外环更多彩色（跟 Google 主视觉一致）
                accent_chance = 0.30 + 0.55 * r_norm
                if self._rand() < accent_chance and self.accent_palette:
                    color = self.accent_palette[
                        int(self._rand() * len(self.accent_palette)) % len(self.accent_palette)
                    ]
                else:
                    color = self.neutral_color
                # 粒子大小：scale 0.59 → 1~2.5px（smooth=True 抗锯齿）
                size = (1.0 + self._rand() * 1.2) * self.particles_scale * 2.4
                # 摆动（让环看起来"呼吸"）
                wob_amp = 0.6 + self._rand() * 1.4
                wob_freq = 0.012 + self._rand() * 0.02
                p = {
                    "ring": ring,
                    "base_angle": base_angle,
                    "r": r,
                    "x": 0.0, "y": 0.0,
                    "color": color,
                    "size": size,
                    "rot_speed": speed * (0.85 + self._rand() * 0.30),
                    "wob_phase": self._rand() * 6.283,
                    "wob_freq": wob_freq,
                    "wob_amp": wob_amp,
                }
                self.particles.append(p)
                it = self.canvas.create_oval(
                    -size, -size, size, size,
                    fill=color, outline="", tags=("dot",),
                )
                self._dot_items.append(it)
                idx += 1
        # 启动 tick
        if self._tick_handle is None:
            self._tick_handle = self.canvas.after(33, self.tick)

    def reinit(self) -> None:
        if self._tick_handle is not None:
            try:
                self.canvas.after_cancel(self._tick_handle)
            except tk.TclError:
                pass
            self._tick_handle = None
        self._init()

    def set_mouse(self, x: float, y: float) -> None:
        self.mouse_x = float(x)
        self.mouse_y = float(y)

    def tick(self) -> None:
        if not self.particles:
            return
        t = time.time()
        cx, cy = self.cx, self.cy
        mr = self.mouse_radius
        mf = self.mouse_force
        for i, p in enumerate(self.particles):
            # 环旋转（每环不同速度）
            angle = p["base_angle"] + t * p["rot_speed"] * 1000  # 缩放到合理速度
            # 摆动（让环有"呼吸感"）
            angle += p["wob_amp"] * 0.005 * math.sin(t * p["wob_freq"] + p["wob_phase"])
            r = p["r"] + p["wob_amp"] * 0.4 * math.cos(t * p["wob_freq"] * 0.7 + p["wob_phase"])
            # 基础位置
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            # 鼠标径向位移（关键特效：环被"压扁"）
            dx = px - self.mouse_x
            dy = py - self.mouse_y
            d2 = dx * dx + dy * dy
            if d2 < mr * mr and d2 > 1.0:
                d = d2 ** 0.5
                # ring-displacement=0.62 → 强位移
                force = (mr - d) * mf / d
                px += dx * force
                py += dy * force
            p["x"], p["y"] = px, py
            # 更新 canvas item
            try:
                self.canvas.coords(
                    self._dot_items[i],
                    px - p["size"], py - p["size"],
                    px + p["size"], py + p["size"],
                )
            except tk.TclError:
                pass
        # 极淡的连接线（让粒子之间有点结构感，但不像 v4.4 那样抢眼）
        self.canvas.delete("line")
        cd = 50
        cd2 = cd * cd
        # 限制每帧最多 250 条线（性能护栏）
        line_budget = 250
        drawn = 0
        for i in range(len(self.particles)):
            if drawn >= line_budget:
                break
            p1 = self.particles[i]
            for j in range(i + 1, len(self.particles)):
                if drawn >= line_budget:
                    break
                p2 = self.particles[j]
                # 只在同环或相邻环画线（性能）
                if abs(p1["ring"] - p2["ring"]) > 1:
                    continue
                dx = p1["x"] - p2["x"]
                dy = p1["y"] - p2["y"]
                d2 = dx * dx + dy * dy
                if d2 < cd2:
                    d = d2 ** 0.5
                    fade = 1.0 - d / cd
                    if self.theme == "light":
                        # 浅灰 → 背景色
                        r = int(170 * fade + 248 * (1 - fade))
                        g = int(170 * fade + 249 * (1 - fade))
                        b = int(170 * fade + 250 * (1 - fade))
                    else:
                        r = int(220 * fade + 20 * (1 - fade))
                        g = int(220 * fade + 20 * (1 - fade))
                        b = int(220 * fade + 30 * (1 - fade))
                    color = f"#{r:02X}{g:02X}{b:02X}"
                    self.canvas.create_line(
                        p1["x"], p1["y"], p2["x"], p2["y"],
                        fill=color, width=0.5, tags=("line",),
                    )
                    drawn += 1
        self._tick_handle = self.canvas.after(33, self.tick)

    def stop(self) -> None:
        if self._tick_handle is not None:
            try:
                self.canvas.after_cancel(self._tick_handle)
            except tk.TclError:
                pass
            self._tick_handle = None


# ===========================================================================
# GUI 主类
# ===========================================================================

_BaseTk = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk


class ImageUpscalerApp(_BaseTk):

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} · AI 超分辨率画质修复")
        self.geometry("900x840")
        self.minsize(780, 720)
        self.configure(bg=COLORS["bg"])

        # 顶部羽毛笔图标
        try:
            icon_path = get_app_root_dir() / "app_icon.ico"
            if icon_path.is_file():
                self.iconbitmap(str(icon_path))
        except tk.TclError:
            pass

        # 加载持久化配置
        self._config = load_config()
        self._config_save_job: Optional[str] = None
        self._restoring_config = False

        # 状态
        self._log_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._processing = False
        self._selected_files: List[Path] = []
        self._drop_hover = False
        self._destroyed = False
        # 粒子系统状态（ParticleMesh 自身管位置 / 速度）

        # 背景画布（最底层，放粒子网格）
        self._bg = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0, bd=0)
        self._bg.place(x=0, y=0, relwidth=1, relheight=1)
        self._bg.tk.call("lower", self._bg._w)

        # 启动期检测
        self._engine_path: Optional[str] = find_engine_path()
        self._onnx_path: Optional[str] = find_onnx_model_path()
        self._msvcp_ok = check_msvcp140_available()

        self._build_styles()
        self._build_ui()

        # 粒子网格背景（Google Antigravity 风格：在 _bg 画布上画，卡片之下）
        self._init_particles()
        self._bg.bind("<Configure>", self._on_bg_configure)

        self._wire_config_traces()
        self._auto_detect_exe()
        self._restore_last_session()

        # 事件循环
        self.bind("<Motion>", self._on_mouse_move)
        self.after(80, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 启动后延迟做一次依赖体检
        self.after(300, self._post_startup_dependency_check)
        # 启动粒子 tick（先等画布首次 layout 完再起）
        self.after(80, self._start_particles)

    # ----------------------------------------------------------------
    # 粒子网格背景（Google Antigravity 风格）
    # ----------------------------------------------------------------

    def _on_bg_configure(self, _event) -> None:
        """_bg 画布尺寸变化时（窗口 resize）重新分布粒子。"""
        if self._destroyed:
            return
        if hasattr(self, "_particles") and self._particles is not None:
            self._particles.reinit()

    def _on_mouse_move(self, event) -> None:
        """鼠标移动 → 更新粒子系统的排斥源位置。"""
        if self._destroyed:
            return
        if hasattr(self, "_particles") and self._particles is not None:
            # 把 toplevel 坐标换算到 _bg 画布坐标
            try:
                bx = event.x_root - self._bg.winfo_rootx()
                by = event.y_root - self._bg.winfo_rooty()
                self._particles.set_mouse(bx, by)
            except tk.TclError:
                pass

    def _init_particles(self) -> None:
        """Google Antigravity 同心环粒子（直接照搬 HTML data-* 参数）。"""
        if self._destroyed:
            return
        self._particles = AntigravityParticles(
            self._bg,
            theme="light",                # 极简白主题
            density=230,                  # ← data-density="230" 一致
            ring_count=5,
            particles_scale=0.59,         # ← data-particles-scale="0.59" 一致
            ring_displacement=0.62,       # ← data-ring-displacement="0.62" 一致
            mouse_radius=180,
        )

    def _start_particles(self) -> None:
        if hasattr(self, "_particles") and self._particles:
            self._particles.tick()

    # ----------------------------------------------------------------
    # 样式
    # ----------------------------------------------------------------

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["card"])

        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["text_dim"],
                        font=("Microsoft YaHei UI", 10))
        style.configure("Version.TLabel", background=COLORS["bg"], foreground=COLORS["text_faint"],
                        font=("Microsoft YaHei UI", 9))
        style.configure("Hint.TLabel", background=COLORS["bg"], foreground=COLORS["text_dim"],
                        font=("Microsoft YaHei UI", 9))
        style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10))
        style.configure("CardTitle.TLabel", background=COLORS["card"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("CardCount.TLabel", background=COLORS["card"], foreground=COLORS["text_dim"],
                        font=("Microsoft YaHei UI", 9))
        style.configure("CardHint.TLabel", background=COLORS["card"], foreground=COLORS["text_faint"],
                        font=("Microsoft YaHei UI", 9))

        style.configure("TEntry", fieldbackground=COLORS["card_alt"],
                        foreground=COLORS["text"], insertcolor=COLORS["text"],
                        bordercolor=COLORS["border"], lightcolor=COLORS["border"],
                        darkcolor=COLORS["border"], padding=8)

        style.configure("TButton", background=COLORS["card"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10), padding=(16, 8),
                        borderwidth=1, relief="solid", bordercolor=COLORS["border_strong"])
        style.map("TButton",
                  background=[("active", COLORS["card_alt"]),
                              ("disabled", COLORS["card_alt"])],
                  foreground=[("disabled", COLORS["text_faint"])],
                  bordercolor=[("active", COLORS["text_faint"])])

        style.configure("Primary.TButton", background=COLORS["primary"],
                        foreground=COLORS["primary_text"],
                        font=("Microsoft YaHei UI", 11, "bold"),
                        padding=(28, 12), borderwidth=0, relief="flat")
        style.map("Primary.TButton",
                  background=[("active", COLORS["primary_hover"]),
                              ("disabled", COLORS["text_faint"])],
                  foreground=[("disabled", COLORS["card"])])

        style.configure("TRadiobutton", background=COLORS["card"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10), focuscolor=COLORS["card"])
        style.map("TRadiobutton",
                  background=[("active", COLORS["card"])],
                  indicatorcolor=[("selected", COLORS["text"]),
                                  ("!selected", COLORS["border_strong"])])

        style.configure("Slim.Horizontal.TProgressbar", troughcolor=COLORS["card_alt"],
                        background=COLORS["text"], bordercolor=COLORS["border"],
                        lightcolor=COLORS["text"], darkcolor=COLORS["text"], thickness=6)

        style.configure("TCombobox", fieldbackground=COLORS["card_alt"],
                        background=COLORS["card"], foreground=COLORS["text"],
                        bordercolor=COLORS["border"], lightcolor=COLORS["border"],
                        darkcolor=COLORS["border"], arrowcolor=COLORS["text_dim"],
                        padding=6, selectbackground=COLORS["card_alt"],
                        selectforeground=COLORS["text"])
        style.map("TCombobox",
                  fieldbackground=[("readonly", COLORS["card_alt"]),
                                   ("disabled", COLORS["card_alt"])],
                  foreground=[("readonly", COLORS["text"])],
                  selectbackground=[("readonly", COLORS["card_alt"])],
                  selectforeground=[("readonly", COLORS["text"])])

    # ----------------------------------------------------------------
    # 界面构建
    # ----------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=4)
        self.rowconfigure(4, weight=3)

        # Row 0 · Header
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(28, 8))
        head_left = ttk.Frame(header)
        head_left.pack(side="left", fill="x", expand=True)
        ttk.Label(head_left, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            head_left,
            text="Real-ESRGAN · 多文件直选 · Vulkan GPU + ONNX CPU 双引擎",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(header, text=APP_VERSION, style="Version.TLabel").pack(
            side="right", anchor="ne", pady=(8, 0)
        )

        # Row 1 · 设置卡片
        settings_w = self._make_card(self)
        settings_w.grid(row=1, column=0, sticky="ew", padx=PAD, pady=(8, 10))
        s = ttk.Frame(settings_w.card, style="Card.TFrame")
        s.pack(fill="x", padx=20, pady=(16, 18))

        self.exe_var = tk.StringVar()
        self._add_path_row(s, "放大引擎", self.exe_var, self._browse_exe, "浏览")

        self.output_var = tk.StringVar()
        self._add_path_row(s, "输出目录", self.output_var, self._browse_output, "选择")

        scale_row = ttk.Frame(s, style="Card.TFrame")
        scale_row.pack(fill="x", pady=(10, 2))
        ttk.Label(scale_row, text="放大倍率/输出档位", style="Card.TLabel", width=14).pack(side="left", anchor="n")

        # 分两行：左 = 倍率 2x/3x/4x（保留兼容）；右 = 目标档位 FHD/2K/4K（新增）
        scale_left = ttk.Frame(scale_row, style="Card.TFrame")
        scale_left.pack(side="left", padx=(0, 20))
        scale_right = ttk.Frame(scale_row, style="Card.TFrame")
        scale_right.pack(side="left")

        ttk.Label(scale_left, text="倍率", style="CardHint.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(scale_right, text="目标档位（长边像素）", style="CardHint.TLabel").pack(anchor="w", pady=(0, 2))

        self.scale_var = tk.StringVar(value="4x")
        for opt in ("2x", "3x", "4x"):
            ttk.Radiobutton(
                scale_left, text=opt, value=opt, variable=self.scale_var
            ).pack(side="left", padx=(0, 12))
        for opt in ("FHD", "2K", "4K"):
            label = f"{opt}（{TARGET_PRESETS[opt]}px）"
            ttk.Radiobutton(
                scale_right, text=label, value=opt, variable=self.scale_var
            ).pack(side="left", padx=(0, 10))

        model_row = ttk.Frame(s, style="Card.TFrame")
        model_row.pack(fill="x", pady=(10, 2))
        ttk.Label(model_row, text="AI 模型", style="Card.TLabel", width=10).pack(side="left")
        self.model_var = tk.StringVar(value="realesrgan-x4plus")
        ttk.Combobox(
            model_row, textvariable=self.model_var, state="readonly",
            # 只列 engine/models/ 里实际存在的模型（x2plus/anime-6B 未随包发布，
            # 选了会 _wfopen 失败）
            values=list(VULKAN_SHIPPED_MODELS),
            width=30,
        ).pack(side="left", fill="x", expand=True)

        # Row 2 · 选择 + 文件列表
        sel_w = self._make_card(self)
        sel_w.grid(row=2, column=0, sticky="nsew", padx=PAD, pady=(0, 10))
        sel = ttk.Frame(sel_w.card, style="Card.TFrame")
        sel.pack(fill="both", expand=True, padx=2, pady=2)

        self.drop_canvas = tk.Canvas(
            sel, height=150, bg=COLORS["card"],
            highlightthickness=0, bd=0, cursor="hand2",
        )
        self.drop_canvas.pack(fill="x", padx=20, pady=(20, 14))
        self.drop_canvas.bind("<Configure>", lambda e: self._redraw_drop())
        self.drop_canvas.bind("<Button-1>", lambda e: self._browse_files())
        if DND_AVAILABLE:
            self.drop_canvas.drop_target_register(DND_FILES)
            self.drop_canvas.dnd_bind("<<Drop>>", self._on_drop_files)
            self.drop_canvas.dnd_bind("<<DropEnter>>", lambda e: self._set_drop_hover(True))
            self.drop_canvas.dnd_bind("<<DropLeave>>", lambda e: self._set_drop_hover(False))

        btn_row = ttk.Frame(sel, style="Card.TFrame")
        btn_row.pack(fill="x", padx=20, pady=(0, 10))
        ttk.Button(btn_row, text="📄  选择图片文件", command=self._browse_files).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="📁  选择文件夹", command=self._browse_folder).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="清空", command=self._clear_files).pack(side="left", padx=(0, 16))
        self.file_summary_var = tk.StringVar(value="尚未选择图片")
        ttk.Label(btn_row, textvariable=self.file_summary_var, style="CardCount.TLabel").pack(side="right")

        list_outer = tk.Frame(sel, bg=COLORS["border"])
        list_outer.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.file_listbox = tk.Listbox(
            list_outer, bg=COLORS["card"], fg=COLORS["text"],
            selectbackground=COLORS["card_alt"], selectforeground=COLORS["text"],
            relief="flat", bd=0, highlightthickness=0,
            font=("Microsoft YaHei UI", 9), activestyle="none",
        )
        list_scroll = ttk.Scrollbar(list_outer, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=list_scroll.set)
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=(1, 0), pady=1)
        list_scroll.pack(side="right", fill="y", pady=1, padx=(0, 1))
        self.file_listbox.bind("<Delete>", self._on_listbox_delete)
        self._set_listbox_empty_hint()

        # Row 3 · 控制按钮
        ctrl = ttk.Frame(self)
        ctrl.grid(row=3, column=0, sticky="ew", padx=PAD, pady=(2, 10))
        self.start_btn = ttk.Button(
            ctrl, text="开始 AI 无损画质放大", style="Primary.TButton",
            command=self._start_processing,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.cancel_btn = ttk.Button(
            ctrl, text="终止任务", command=self._cancel_processing, state="disabled",
        )
        self.cancel_btn.pack(side="right")

        # Row 4 · 日志卡片
        log_w = self._make_card(self)
        log_w.grid(row=4, column=0, sticky="nsew", padx=PAD, pady=(0, 24))
        log_inner = ttk.Frame(log_w.card, style="Card.TFrame")
        log_inner.pack(fill="both", expand=True, padx=20, pady=(14, 18))

        log_header = ttk.Frame(log_inner, style="Card.TFrame")
        log_header.pack(fill="x", pady=(0, 8))
        ttk.Label(log_header, text="处理日志", style="CardTitle.TLabel").pack(side="left")
        ttk.Button(log_header, text="清空", command=self._clear_log).pack(side="right")

        self.progress = ttk.Progressbar(
            log_inner, style="Slim.Horizontal.TProgressbar",
            mode="determinate", maximum=100,
        )
        self.progress.pack(fill="x", pady=(0, 8))

        console_frame = tk.Frame(log_inner, bg=COLORS["border"], bd=0)
        console_frame.pack(fill="both", expand=True)
        self.console = tk.Text(
            console_frame, wrap="word", height=6,
            bg=COLORS["card_alt"], fg=COLORS["text"],
            insertbackground=COLORS["text"],
            font=("Consolas", 10), relief="flat", bd=6, state="disabled",
        )
        scrollbar = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)
        self.console.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.console.tag_configure("info", foreground=COLORS["text"])
        self.console.tag_configure("success", foreground=COLORS["success"])
        self.console.tag_configure("error", foreground=COLORS["danger"])
        self.console.tag_configure("warning", foreground=COLORS["warning"])

        self._append_log("软件初始化完成。请选择图片文件或拖入开始。", "info")

    # ----------------------------------------------------------------
    # 卡片 helper
    # ----------------------------------------------------------------

    def _make_card(self, parent) -> tk.Frame:
        wrapper = tk.Frame(parent, bg=COLORS["border"], highlightthickness=0, bd=0)
        card = ttk.Frame(wrapper, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=1, pady=1)
        wrapper.card = card  # type: ignore[attr-defined]
        return wrapper

    def _add_path_row(self, parent, label, variable, browse_cmd, browse_text) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, style="Card.TLabel", width=10).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(row, text=browse_text, command=browse_cmd, width=8).pack(side="right")

    # ----------------------------------------------------------------
    # 拖拽区
    # ----------------------------------------------------------------

    def _redraw_drop(self) -> None:
        c = self.drop_canvas
        c.delete("all")
        w = max(c.winfo_width(), 1)
        h = max(c.winfo_height(), 1)
        border_color = COLORS["text"] if self._drop_hover else COLORS["border_strong"]
        pad = 2
        c.create_rectangle(
            pad, pad, w - pad, h - pad,
            outline=border_color, dash=(8, 5), width=1,
        )
        c.create_text(
            w / 2, h / 2 - 18,
            text="⬇  点击此处选择图片，或将文件 / 文件夹拖入此区域",
            font=("Microsoft YaHei UI", 13, "bold"),
            fill=COLORS["text"],
        )
        c.create_text(
            w / 2, h / 2 + 8,
            text="支持 PNG · JPG · JPEG · WEBP · BMP · TIFF",
            font=("Microsoft YaHei UI", 10),
            fill=COLORS["text_dim"],
        )
        c.create_text(
            w / 2, h / 2 + 30,
            text="拖入文件夹将自动递归扫描所有有效图片",
            font=("Microsoft YaHei UI", 9),
            fill=COLORS["text_faint"],
        )

    def _set_drop_hover(self, on: bool) -> None:
        self._drop_hover = on
        self._redraw_drop()

    # ----------------------------------------------------------------
    # 文件选择 / 拖放（含非图片过滤弹窗）
    # ----------------------------------------------------------------

    def _browse_files(self) -> None:
        # 强制单选：一次只处理一张，避免多图任务被后台偷偷并发跑
        path = filedialog.askopenfilename(
            title="选择单张图片（一次仅处理一张）",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("PNG", "*.png"), ("JPG/JPEG", "*.jpg *.jpeg"),
                ("WEBP", "*.webp"), ("BMP", "*.bmp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        new_files = [Path(path)] if Path(path).is_file() else []
        if new_files:
            # 单选模式：替换而非追加，避免列表里堆积历史图片
            self._selected_files.clear()
            self._add_files(new_files)
            self._set_last_opened_path(new_files[0].parent)
            self._append_log(f"已选择图片：{new_files[0].name}", "info")

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="选择图片文件夹（将递归扫描）")
        if not path:
            return
        self._add_folder(Path(path))
        self._set_last_opened_path(Path(path))

    def _on_drop_files(self, event) -> None:
        try:
            raw = str(getattr(event, "data", "") or "")
            items = self.tk.splitlist(raw) if raw else []
        except Exception:  # noqa: BLE001
            items = []
        if not items:
            return

        new_files: List[Path] = []
        first_dir: Optional[Path] = None
        total_files = 0
        total_dirs = 0
        rejected: List[str] = []

        for raw_path in items:
            p = Path(raw_path)
            if p.is_file():
                total_files += 1
                if is_image_path(p):
                    new_files.append(p)
                    if first_dir is None:
                        first_dir = p.parent
                else:
                    rejected.append(p.name)
            elif p.is_dir():
                total_dirs += 1
                imgs = collect_images(p)
                if imgs:
                    new_files.extend(imgs)
                    if first_dir is None:
                        first_dir = p
                else:
                    rejected.append(f"{p.name}/ (空文件夹)")

        # 过滤提示：拖入了非图片格式 → 弹窗告知
        if rejected:
            preview = "、".join(rejected[:3]) + ("…" if len(rejected) > 3 else "")
            messagebox.showinfo(
                "已自动过滤非图片格式",
                f"已自动过滤非图片格式，仅保留 PNG/JPG/WEBP 等图像。\n"
                f"被过滤项：{preview}",
            )

        if new_files:
            self._add_files(new_files)
            if first_dir is not None:
                self._set_last_opened_path(first_dir)
            self._append_log(
                f"拖放导入 {len(new_files)} 张图片"
                + (f"（{total_files} 文件 + {total_dirs} 文件夹）" if total_dirs else ""),
                "info",
            )
        else:
            self._append_log("拖入内容中没有可处理的图片文件。", "warning")

    def _add_files(self, files: List[Path]) -> None:
        existing = {str(p.resolve()) for p in self._selected_files}
        added = 0
        for f in files:
            try:
                key = str(f.resolve())
            except OSError:
                key = str(f)
            if key not in existing:
                self._selected_files.append(f)
                existing.add(key)
                added += 1
        if added:
            self._update_summary()
            self._refresh_file_list()
            if not self.output_var.get().strip() and self._selected_files:
                self.output_var.set(default_output_for(self._selected_files))
            # 智能模型推荐：仅在用户未手动改过模型时自动切换
            self._auto_pick_model_if_default(files)

    def _auto_pick_model_if_default(self, files: List[Path]) -> None:
        """第一次导入文件时按路径关键词智能选模型；
        用户已手动改过模型（model_var 不再是默认）则不再骚扰。
        """
        try:
            current = self.model_var.get()
        except tk.TclError:
            return
        # 默认值列表：用户从未改过模型时才会触发自动推荐
        defaults = {"realesrgan-x4plus"}
        if current not in defaults:
            return
        best = self._pick_best_model(files)
        if best != current:
            # 用 _restoring_config=True 短暂抑制 trace_add 触发写盘
            self._restoring_config = True
            try:
                self.model_var.set(best)
            finally:
                self._restoring_config = False
            self._append_log(
                f"🪄 已根据文件路径智能推荐 AI 模型：{current} → {best}", "info"
            )

    def _add_folder(self, folder: Path) -> None:
        images = collect_images(folder)
        if not images:
            self._append_log(f"文件夹 {folder} 中没有找到图片。", "warning")
            return
        self._add_files(images)
        self._append_log(f"从 {folder.name}/ 导入 {len(images)} 张图片", "info")

    def _clear_files(self) -> None:
        if not self._selected_files:
            return
        n = len(self._selected_files)
        self._selected_files.clear()
        self._update_summary()
        self._refresh_file_list()
        self._append_log(f"已清空 {n} 个待处理文件", "info")

    def _on_listbox_delete(self, _event) -> None:
        sel = list(self.file_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            if 0 <= idx < len(self._selected_files):
                del self._selected_files[idx]
        self._update_summary()
        self._refresh_file_list()

    # ----------------------------------------------------------------
    # 文件列表 / 摘要
    # ----------------------------------------------------------------

    def _set_listbox_empty_hint(self) -> None:
        self.file_listbox.delete(0, "end")
        self.file_listbox.insert("end", "  尚未选择图片 · 点击上方区域选择 / 拖入")

    def _refresh_file_list(self) -> None:
        self.file_listbox.delete(0, "end")
        if not self._selected_files:
            self._set_listbox_empty_hint()
            return
        show = self._selected_files[:FILE_LIST_MAX_SHOW]
        for p in show:
            label = self._display_name_for(p)
            self.file_listbox.insert("end", f"  {label}")
        if len(self._selected_files) > FILE_LIST_MAX_SHOW:
            self.file_listbox.insert(
                "end",
                f"  … 其余 {len(self._selected_files) - FILE_LIST_MAX_SHOW} 张已折叠（处理时仍会全部执行）",
            )

    def _display_name_for(self, p: Path) -> str:
        same_name = sum(
            1 for q in self._selected_files
            if q.name == p.name and q.parent != p.parent
        )
        if same_name == 0:
            return p.name
        return f"{p.parent.name}{os.sep}{p.name}"

    def _update_summary(self) -> None:
        n = len(self._selected_files)
        if n == 0:
            self.file_summary_var.set("尚未选择图片")
            return
        total = 0
        for f in self._selected_files:
            try:
                total += f.stat().st_size
            except OSError:
                pass
        self.file_summary_var.set(f"已选 {n} 张  ·  {format_size(total)}")

    # ----------------------------------------------------------------
    # 配置持久化
    # ----------------------------------------------------------------

    def _wire_config_traces(self) -> None:
        self.scale_var.trace_add("write", lambda *_: self._on_config_change())
        self.model_var.trace_add("write", lambda *_: self._on_config_change())

    def _on_config_change(self) -> None:
        if self._restoring_config:
            return
        self._schedule_save_config()

    def _schedule_save_config(self) -> None:
        if self._destroyed:
            return
        if self._config_save_job:
            try:
                self.after_cancel(self._config_save_job)
            except Exception:  # noqa: BLE001
                pass
        self._config_save_job = self.after(500, self._save_config_now)

    def _save_config_now(self) -> None:
        self._config_save_job = None
        cfg = dict(self._config)
        cfg["scale_factor"] = self.scale_var.get() or DEFAULT_CONFIG["scale_factor"]
        cfg["model_name"] = self.model_var.get() or DEFAULT_CONFIG["model_name"]
        if save_config(cfg):
            self._config = cfg

    def _set_last_opened_path(self, path: Path) -> None:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        self._config["last_opened_path"] = resolved
        self._schedule_save_config()

    def _restore_last_session(self) -> None:
        cfg = self._config
        self._restoring_config = True
        try:
            scale = cfg.get("scale_factor") or DEFAULT_CONFIG["scale_factor"]
            if scale in SCALE_OPTIONS:
                self.scale_var.set(scale)
            model = cfg.get("model_name") or DEFAULT_CONFIG["model_name"]
            # 旧配置可能存着未随包发布的模型名 → 回落到默认，防止 _wfopen 失败
            if model not in VULKAN_SHIPPED_MODELS:
                model = DEFAULT_CONFIG["model_name"]
            if model:
                self.model_var.set(model)
        finally:
            self._restoring_config = False

        last = (cfg.get("last_opened_path") or "").strip()
        if not last:
            return
        p = Path(last)
        if not p.exists():
            self._append_log(f"上次的路径已不存在：{last}", "info")
            return
        if p.is_dir():
            images = collect_images(p)
            if images:
                self._add_files(images)
                self._append_log(f"已恢复上次会话：{p}（{len(images)} 张图片）", "info")
            else:
                self._append_log(f"上次路径 {p} 中未发现图片", "info")
        elif p.is_file():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                self._add_files([p])
                self._append_log(f"已恢复上次会话：{p.name}", "info")
            else:
                self._append_log(f"上次文件 {p.name} 不是图片格式", "info")

    # ----------------------------------------------------------------
    # 路径浏览 + 引擎探测
    # ----------------------------------------------------------------

    def _auto_detect_exe(self) -> None:
        if self._engine_path:
            # 美化显示：内置引擎（在 _MEIPASS 中）→ 只显示 basename + "内置" 标签
            display = self._friendly_engine_label(self._engine_path)
            self.exe_var.set(display)  # 输入框显示友好标签
            # 真实路径保留在 self._engine_path（_start_processing 会用）
            self._append_log(f"已自动定位本地引擎：{display}", "success")
        else:
            root = get_app_root_dir()
            engine_dir = root / "engine"
            try:
                engine_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._append_log(f"创建 engine 目录失败：{exc}", "error")
            self._append_log("未在应用根目录找到 Vulkan 引擎，请手动选择。", "warning")
            self.after(200, self._show_missing_engine_dialog)

        if self._onnx_path:
            self._append_log(f"已定位 ONNX 保底模型：{self._onnx_path}", "success")
        else:
            self._append_log(
                "未在 models/ 找到 ONNX 模型（Vulkan 引擎失败时无法自动降级）",
                "warning",
            )

        # 新增：Vulkan 引擎完整性体检（EXE 在但模型文件缺失是最常见的坑）
        self._check_vulkan_models_integrity()

    def _check_vulkan_models_integrity(self) -> None:
        """检测 engine/models/ 下的 ncnn 模型文件是否完整。

        典型坑：release zip 只包含 EXE/DLL，不包含 models/，导致
        "_wfopen ... models/realesrgan-x4plus.param failed"。
        解决方案：把 .param 和 .bin 放入 engine/models/。
        """
        if not self._engine_path:
            return  # EXE 都没有，不做模型检查
        # 找引擎所在目录下的 models/ 子目录
        try:
            engine_dir = Path(self._engine_path).resolve().parent
        except OSError:
            return
        models_dir = engine_dir / "models"
        # 至少要有 realesrgan-x4plus.param/.bin
        must_have = models_dir / "realesrgan-x4plus.param"
        if must_have.is_file() and must_have.stat().st_size > 1024:
            self._append_log(
                f"已检测到 Vulkan 模型：{models_dir}",
                "success",
            )
            return

        # 缺失 → 写一条 ERROR 级日志 + 弹窗，附上明确修复路径
        msg = (
            "Vulkan 引擎可执行文件已就位，但 models/ 目录下缺少 ncnn 模型文件 "
            "(realesrgan-x4plus.param / .bin)。\n\n"
            f"引擎目录：{engine_dir}\n"
            f"模型目录：{models_dir}\n\n"
            "请从 Real-ESRGAN 官方 release 下载包含 models/ 的完整包，"
            "把 models 文件夹内的 .param / .bin 全部复制到上述模型目录后重启应用。\n\n"
            "推荐来源：\n"
            "  https://github.com/xinntao/Real-ESRGAN/releases\n"
            "  （realesrgan-ncnn-vulkan-20220424-windows.zip）"
        )
        self._append_log("Vulkan 引擎缺少模型文件（.param/.bin）", "error")
        self._append_log(f"模型目录：{models_dir}", "error")
        if not self._destroyed:
            self.after(300, lambda: messagebox.showerror("Vulkan 模型缺失", msg))

    def _show_missing_engine_dialog(self) -> None:
        if self._destroyed:
            return
        root = get_app_root_dir()
        messagebox.showwarning(
            "未找到 Vulkan 引擎",
            f"未在应用根目录找到 realesrgan-ncnn-vulkan.exe\n\n"
            f"应用根目录：{root}\n"
            f"已自动创建子目录：{root / 'engine'}\n\n"
            f"请将引擎可执行文件及其依赖 DLL 放入该目录后重启，"
            f"或点击「浏览」手动指定可执行文件路径。",
        )

    @staticmethod
    def _friendly_engine_label(path: str) -> str:
        """把引擎真实路径转成人类可读的短标签。
        - 位于 PyInstaller _MEIPASS（内置引擎）→ "内置引擎 · realesrgan-ncnn-vulkan.exe"
        - 位于应用根目录的 engine/ → "根目录引擎 · engine\\realesrgan-ncnn-vulkan.exe"
        - 其它 → 原始路径
        """
        if not path:
            return ""
        norm = path.replace("/", os.sep)
        # 1) PyInstaller 内置（注意：meipass 可能是 / 或 \\，统一归一化）
        meipass_raw = getattr(sys, "_MEIPASS", None)
        meipass_norm = meipass_raw.replace("/", os.sep) if meipass_raw else ""
        if meipass_norm and norm.lower().startswith(meipass_norm.lower()):
            return f"内置引擎 · {Path(path).name}"
        # 2) 应用根目录的 engine/
        try:
            root = get_app_root_dir()
            try:
                rel = Path(path).resolve().relative_to(root.resolve())
                return f"根目录引擎 · {rel}"
            except (ValueError, OSError):
                pass
        except OSError:
            pass
        # 3) 兜底：原路径
        return path

    @staticmethod
    def _pick_best_model(files: List[Path]) -> str:
        """根据文件路径关键词智能推荐 AI 模型。

        只返回 VULKAN_SHIPPED_MODELS 里的名字（未随包发布的模型选了必炸）。

        启发式：
          - 路径含 "anime" / "动漫" / "二次元" → realesrgan-x4plus-anime
          - 其它 → realesrgan-x4plus
        """
        if not files:
            return "realesrgan-x4plus"
        joined = " ".join(str(f).lower() for f in files[:32])
        if any(k in joined for k in ("anime", "动漫", "二次元", "cartoon", "漫画")):
            return "realesrgan-x4plus-anime"
        return "realesrgan-x4plus"

    def _browse_exe(self) -> None:
        filetypes = (
            [("可执行文件", "*.exe"), ("所有文件", "*.*")]
            if os.name == "nt"
            else [("所有文件", "*.*")]
        )
        path = filedialog.askopenfilename(
            title="选择 realesrgan-ncnn-vulkan 引擎", filetypes=filetypes
        )
        if path:
            # 同样美化显示：用户手动选的直接显示路径（已经是干净的）
            self._engine_path = path
            self.exe_var.set(self._friendly_engine_label(path))
            self._append_log(f"已手动选择引擎：{path}", "info")

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="选择放大图片的导出目录")
        if path:
            self.output_var.set(path)

    # ----------------------------------------------------------------
    # 启动后依赖体检（MSVCP140 / onnxruntime / 引擎完整性）
    # ----------------------------------------------------------------

    def _post_startup_dependency_check(self) -> None:
        if self._destroyed:
            return
        # 1) MSVCP140.dll 缺失
        if not self._msvcp_ok and self._engine_path:
            self._append_log(
                "系统缺少 MSVCP140.dll（VC++ 运行库），Vulkan 引擎可能无法启动。",
                "error",
            )
            self._show_toast(
                "系统缺少 C++ 运行库组件（MSVCP140.dll），请安装 VC++ "
                "Redistributable 后重试。",
                "error", duration_ms=5200,
            )
            return
        # 2) onnxruntime 缺失但用户想用 ONNX
        if self._onnx_path and not ONNX_RUNTIME_OK:
            self._append_log(
                "检测到 ONNX 模型但未安装 onnxruntime，CPU 兜底模式不可用。",
                "warning",
            )
        # 3) Vulkan + ONNX 双双缺失
        if not self._engine_path and not self._onnx_path:
            self._append_log(
                "未找到任何可用引擎！请放置 Vulkan 可执行文件或 ONNX 模型。",
                "error",
            )

    # ----------------------------------------------------------------
    # 任务调度
    # ----------------------------------------------------------------

    def _validate(self) -> Optional[str]:
        # 用真实路径校验（exe_var 现在显示的是友好标签）
        exe = self._engine_path or ""
        if exe and not Path(exe).is_file():
            return "引擎路径无效，请重新选择。"
        # 至少要有一个引擎
        if not exe and not self._onnx_path:
            return "未找到任何可用引擎（Vulkan / ONNX 都不可用）。"
        if not self._selected_files:
            return "请先选择至少一张待处理的图片（文件或文件夹）。"
        # 模型文件存在性校验：选了没随包发布的模型 → 提前拦截（比 _wfopen 报错友好）
        if exe:
            try:
                model_name = self.model_var.get()
            except tk.TclError:
                model_name = ""
            if model_name and model_name not in VULKAN_SHIPPED_MODELS:
                return f"模型 {model_name} 未随本应用发布，请在下拉框中选择可用模型。"
            models_dir = find_vulkan_models_dir(exe)
            if models_dir and model_name:
                probe = models_dir / f"{model_name}.param"
                if not probe.is_file():
                    return f"模型 {model_name} 缺少文件（{probe} 不存在），请重新选择模型。"
        output_dir = self.output_var.get().strip() or default_output_for(self._selected_files)
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"无法创建导出目录：{exc}"
        return None

    def _check_high_resolution_warning(self) -> None:
        """CPU 模式 + 大图 → 弹气泡。FHD/2K/4K 档位不参与。"""
        scale_text = self.scale_var.get()
        if scale_text in TARGET_PRESETS:
            return  # 目标档位不适用倍率预警
        try:
            scale = int(scale_text.replace("x", ""))
        except (ValueError, AttributeError):
            return
        # 仅 ONNX 模式且图片较大时预警
        max_px = 0
        for f in self._selected_files:
            try:
                if PIL_OK:
                    with Image.open(f) as img:
                        max_px = max(max_px, max(img.size))
            except Exception:  # noqa: BLE001
                continue
        if max_px <= ONNX_RESOLUTION_WARN_PX:
            return
        # 4x + ONNX 唯一可用路径 → 必然高负载
        if scale == 4 and self._onnx_path and (not self._engine_path or not self._msvcp_ok):
            self._show_toast(
                "当前为 CPU 模式且图片分辨率较高，已自动启用内存优化切片，"
                "处理需要一点时间，请耐心等待...",
                "warning", duration_ms=4800,
            )

    def _start_processing(self) -> None:
        if self._processing:
            return
        error = self._validate()
        if error:
            # 特定误操作 → 红色气泡；其它 → 日志 + 弹窗
            if "请先选择" in error:
                self._show_toast("请先拖入或选择需要处理的图片！", "error", duration_ms=2800)
            else:
                self._append_log(error, "error")
                messagebox.showwarning("无法开始", error)
            return

        # CPU 模式高分辨率预警
        self._check_high_resolution_warning()

        output_dir = Path(
            self.output_var.get().strip() or default_output_for(self._selected_files)
        )
        self.output_var.set(str(output_dir))
        scale_text = self.scale_var.get()
        target_preset = ""
        if scale_text in TARGET_PRESETS:
            target_preset = scale_text
            scale = 4  # 内部分辨率档位的 Vulkan 步
        else:
            try:
                scale = int(scale_text.replace("x", ""))
            except (ValueError, AttributeError):
                scale = 4

        files_snapshot = list(self._selected_files)

        # 单图模式：截断到 1 张，防止用户拖入/扫描了多张时后台偷偷全跑
        if len(files_snapshot) > 1:
            kept = files_snapshot[0]
            extra = len(files_snapshot) - 1
            self._show_toast(
                f"单图模式：仅处理第 1 张「{kept.name}」，其余 {extra} 张已忽略",
                "warning", duration_ms=3600,
            )
            self._append_log(
                f"单图模式触发：{len(files_snapshot)} 张已截断为 1 张（{kept.name}）",
                "warning",
            )
            files_snapshot = [kept]
            # 同步清空 UI 列表，避免视觉残留
            self._selected_files.clear()
            self._selected_files.append(kept)
            self._refresh_file_list()
            self._update_summary()

        self._stop_event.clear()
        self._set_processing(True)
        self.progress.configure(value=0, maximum=100)
        self._append_log("—" * 50, "info")
        self._append_log(f"待处理文件：{len(files_snapshot)} 张", "info")
        self._append_log(f"导出目录：{output_dir}", "info")
        if target_preset:
            self._append_log(
                f"目标档位：{target_preset}（长边 {TARGET_PRESETS[target_preset]}px；"
                f"Vulkan x4 + LANCZOS + USM/CLAHE 软化）",
                "info",
            )
        else:
            self._append_log(f"超分倍数：{scale}x", "info")

        worker = UpscaleWorker(
            engine_path=self._engine_path,  # 用真实路径，不用 exe_var（那是友好标签）
            onnx_model_path=self._onnx_path,
            files=files_snapshot,
            output_dir=output_dir,
            scale=scale,
            model=self.model_var.get(),
            log=self._enqueue_log,
            on_finished=self._on_worker_finished,
            stop_event=self._stop_event,
            on_progress=lambda d, t: self._log_queue.put(
                ("__PROGRESS__", f"{d}/{t}")
            ),
            target_preset=target_preset,
        )
        self._worker_thread = threading.Thread(target=worker.run, daemon=True)
        self._worker_thread.start()

    def _cancel_processing(self) -> None:
        if not self._processing:
            return
        self._stop_event.set()
        self._append_log("正在停止任务...", "warning")
        self.cancel_btn.configure(state="disabled")

    def _on_worker_finished(self, success: bool) -> None:
        self._log_queue.put(("__FINISHED__", "success" if success else "warning"))

    def _set_processing(self, busy: bool) -> None:
        self._processing = busy
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    # ----------------------------------------------------------------
    # Toast 辅助
    # ----------------------------------------------------------------

    def _show_toast(self, message: str, level: str = "info", duration_ms: int = TOAST_DEFAULT_MS) -> None:
        try:
            Toast(self, message, level=level, duration_ms=duration_ms)
        except tk.TclError:
            # 兜底：退化为 messagebox
            messagebox.showinfo("提示", message)

    # ----------------------------------------------------------------
    # 日志 / 进度队列
    # ----------------------------------------------------------------

    def _enqueue_log(self, message: str, level: str = "info") -> None:
        self._log_queue.put((message, level))

    def _poll_log_queue(self) -> None:
        if self._destroyed:
            return
        try:
            while True:
                message, level = self._log_queue.get_nowait()
                if message == "__FINISHED__":
                    self._set_processing(False)
                    if level == "success":
                        messagebox.showinfo("完成", "所有图片超分重构完成！")
                    continue
                if message == "__PROGRESS__":
                    try:
                        d, t = level.split("/")
                        self.progress.configure(maximum=int(t), value=int(d))
                    except (ValueError, TypeError):
                        pass
                    continue
                self._append_log(message, level)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _append_log(self, message: str, level: str = "info") -> None:
        self.console.configure(state="normal")
        self.console.insert("end", message + "\n", level)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_log(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _on_close(self) -> None:
        if self._processing:
            if not messagebox.askyesno("确认退出", "后台正在放大图片，确定要强制退出吗？"):
                return
            self._stop_event.set()
        self._save_config_now()
        self._destroyed = True
        # 停掉粒子 tick
        if hasattr(self, "_particles") and self._particles is not None:
            self._particles.stop()
        self.destroy()


# ===========================================================================
# 入口
# ===========================================================================

def main() -> None:
    app = ImageUpscalerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
