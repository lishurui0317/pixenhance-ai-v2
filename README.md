# 图片无损画质增强大师 (PixEnhance AI)

> **v4.0** — AI 驱动的图片超分辨率放大工具，双引擎智能切换

## 功能特性

- **双引擎无缝切换**：优先使用 Vulkan GPU 加速（real-esrgan-ncnn-vulkan），失败时自动降级到 ONNX CPU 推理
- **多种放大模式**：2x / 3x / 4x 倍率放大，以及 FHD / 2K / 4K 目标分辨率预设
- **智能切片**：大图自动切片处理，防止爆显存
- **拖放支持**：拖拽图片自动过滤非图片格式
- **极简白 UI**：纯黑按钮高亮，带鼠标流体漩涡粒子特效
- **Toast 非模态提示**：不阻塞操作的气泡通知
- **配置持久化**：自动保存上次路径、放大倍数、模型选择

## 使用方式

### 直接运行

```bash
pip install -r requirements.txt
python image_upscaler.py
```

### 打包成 exe

```bash
pip install pyinstaller
pyinstaller 图片无损画质增强大师.spec
```

或使用另一个 spec：

```bash
pyinstaller PixEnhance_AI_画质超分大师.spec
```

## 依赖

- Python 3.10+
- tkinter（Python 内置）
- onnxruntime（可选，CPU 保底引擎）
- numpy
- Pillow
- tkinterdnd2（可选，拖放支持）

GPU 加速需要 `engine/` 目录下的 Vulkan 引擎文件（`realesrgan-ncnn-vulkan.exe` 及相关 DLL）。

## 项目结构

```
pyt/
├── image_upscaler.py          # 主程序入口
├── config.json                # 用户配置持久化
├── convert_pth_to_onnx.py     # 模型转换脚本
├── download_onnx_model.py     # 模型下载脚本
├── generate_sketch_feather_icon.py  # 图标生成
├── gen_icon.py                # 图标生成
├── engine/                    # Vulkan 引擎（需自行下载）
├── models/                    # 模型权重（需自行下载）
├── dist/                      # 打包输出目录
└── .gitignore
```

## 注意事项

- 模型权重文件较大（~67MB），需自行下载放置在 `models/` 目录下
- Vulkan 引擎文件需放置在 `engine/` 目录下
- 首次使用建议运行 `download_onnx_model.py` 下载 ONNX 模型
