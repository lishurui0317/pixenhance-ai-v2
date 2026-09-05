#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 RealESRGAN_x4plus.pth (PyTorch 权重) 转换为 ONNX 格式。
================================================================
依赖安装（需要联网一次）：
  pip install torch onnx

使用方法：
  python convert_pth_to_onnx.py
  # 或指定源 / 目标
  python convert_pth_to_onnx.py --src models/RealESRGAN_x4plus.pth --dst models/realesrgan-x4plus.onnx

输出：
  - models/realesrgan-x4plus.onnx（约 64MB，FP32）

转换完成后，image_upscaler.py 在 Vulkan 引擎不可用时会自动启用 ONNX CPU 兜底。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def get_app_root_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-ESRGAN .pth → .onnx 转换器")
    root = get_app_root_dir()
    parser.add_argument(
        "--src", type=str,
        default=str(root / "models" / "RealESRGAN_x4plus.pth"),
        help="源 .pth 权重路径",
    )
    parser.add_argument(
        "--dst", type=str,
        default=str(root / "models" / "realesrgan-x4plus.onnx"),
        help="目标 .onnx 路径",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本（默认 17）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_file():
        print(f"❌ 源文件不存在：{src}")
        print("   请先执行 download_onnx_model.py 下载 PyTorch 权重。")
        return 1

    if dst.is_file() and dst.stat().st_size > 60_000_000:
        ans = input(f"⚠️ 目标文件已存在 {dst}，是否覆盖？[y/N] ").strip().lower()
        if ans != "y":
            print("已取消。")
            return 0

    # 1) 导入 torch + onnx
    try:
        import torch  # noqa: F401
    except ImportError:
        print("❌ 未安装 torch，请先执行：pip install torch")
        return 1
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("❌ 未安装 onnx，请先执行：pip install onnx")
        return 1
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        print("❌ 未安装 basicsr，请先执行：pip install basicsr")
        return 1

    import torch

    print(f"⏳ 加载权重: {src}")
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4,
                  num_feat=64, num_block=23, num_grow_ch=32)
    state = torch.load(src, map_location="cpu", weights_only=False)
    key = "params_ema" if "params_ema" in state else (
        "params" if "params" in state else None
    )
    if key is None:
        net.load_state_dict(state, strict=True)
    else:
        net.load_state_dict(state[key], strict=True)
    net.eval()
    print("✅ 权重加载完成")

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"⏳ 导出 ONNX: {dst}")
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, str(dst),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "N", 2: "H", 3: "W"},
                "output": {0: "N", 2: "H", 3: "W"},
            },
            opset_version=args.opset,
        )
    sz = dst.stat().st_size
    print(f"✨ 转换完成: {dst} ({sz:,} bytes, {sz/1024/1024:.1f} MB)")
    print("   现在 image_upscaler.py 在 Vulkan 不可用时会自动启用 ONNX CPU 兜底。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
