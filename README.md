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

## 下载安装

### 方式一：直接下载安装包（推荐）

下载 `yupianxiangsu/Windows/installer_output/Yupianxiangsu_v1.0.0_Setup.exe`，双击运行安装向导即可完成安装。

安装完成后：
- 桌面会自动创建快捷方式
- 开始菜单也会有程序入口
- 控制面板支持正常卸载

### 方式二：运行裸 exe

如果不想安装，也可以直接下载 `yupianxiangsu/Windows/dist/Yupianxiangsu.exe`，双击运行。

### 方式三：从源码运行

```bash
pip install -r requirements.txt
python image_upscaler.py
```

### 打包成 exe

```bash
pip install pyinstaller
pyinstaller 图片无损画质增强大师.spec
```

## 项目结构

```
pyt/
├── image_upscaler.py              # 主程序入口
├── config.json                    # 用户配置持久化
├── convert_pth_to_onnx.py         # 模型转换脚本
├── download_onnx_model.py         # 模型下载脚本
├── generate_sketch_feather_icon.py  # 图标生成
├── gen_icon.py                    # 图标生成
├── engine/                        # Vulkan 引擎 + 模型文件
├── models/                        # 模型权重
├── yupianxiangsu/                 # 御·像素完整安装包
│   └── Windows/
│       ├── dist/Yupianxiangsu.exe                # 裸 exe
│       ├── installer_output/Yupianxiangsu_v1.0.0_Setup.exe  # 安装包
│       └── build_installer.iss                   # InnoSetup 安装脚本
├── output/                        # 测试输出
├── .bak/                          # 备份
└── .gitignore
```

## 注意事项

- 模型权重文件较大（~64MB），已包含在仓库中
- 首次使用会自动检测 GPU 驱动，如果没有 Vulkan 支持会降级到 CPU 模式
- 如需自行打包安装包，需要安装 [Inno Setup](https://jrsoftware.org/isinfo.php)