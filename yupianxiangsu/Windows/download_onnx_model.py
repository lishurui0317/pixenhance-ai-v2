"""
Real-ESRGAN ONNX 模型自动下载脚本（嵌入版）：
- 优先尝试 huggingface 镜像
- 失败则尝试 GitHub release
- 全部失败给出明确报错，不静默
"""
import os
import sys
import urllib.request
import socket

CANDIDATE_URLS = [
    "https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/realesrgan-x4plus.onnx",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/realesrgan-x4plus.onnx",
]

TIMEOUT = 30


def _root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__ if "__file__" in globals() else sys.argv[0]))


def download_onnx_model() -> str:
    root = _root()
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "realesrgan-x4plus.onnx")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
        print(f"✅ 模型已存在: {model_path} ({os.path.getsize(model_path):,} bytes)")
        return model_path

    socket.setdefaulttimeout(TIMEOUT)
    last_err = None
    for url in CANDIDATE_URLS:
        print(f"⏳ 尝试下载: {url}")
        try:
            urllib.request.urlretrieve(url, model_path)
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
                print(f"✨ 下载完成: {os.path.abspath(model_path)} ({os.path.getsize(model_path):,} bytes)")
                return model_path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  ⚠️ 该镜像失败: {exc}")
    raise RuntimeError(f"所有镜像均失败: {last_err}")


if __name__ == "__main__":
    download_onnx_model()
