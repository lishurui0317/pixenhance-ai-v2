# 项目长期笔记 · E:\xiaomubiao\pyt

## 项目本体
- **图片无损画质增强大师 / PixEnhance_AI_画质超分大师**（两个名字混用，命令用后者，spec/旧产物用前者）
- 入口：`image_upscaler.py`（**tkinter GUI**，subprocess 调 Real-ESRGAN）—— 不是 PyQt5
- v2.0 调色板（见 `2026-08-30.md` 第三轮）：bg #18181C / surface #22222A / border #33333E / primary #413C4F（钛空灰，无亮蓝）
- 引擎目录：`engine/`（realesrgan-ncnn-vulkan.exe + vcomp140[.d].dll + LICENSE + README.md）

## 关键工具链
- **打包解释器**：`D:\python\python.exe`（Python 3.13.15，PyInstaller 6.22.2，**无 PIL**）
- **图标/PIL 环境**：`C:\Users\Administrator\.workbuddy\binaries\python\envs\default` (Python 3.13.14, Pillow 12.3.0) —— 隔离 venv，**不污染** D:\python
- **打包命令模板**（参见 `2026-08-30.md`）：
  ```
  "D:/python/python.exe" -m PyInstaller --noconsole --onefile \
      --name="PixEnhance_AI_画质超分大师" \
      --icon="app_icon.ico" \
      --add-data "engine;engine" \
      image_upscaler.py
  ```

## 验证套路
1. `pyi-archive_viewer --list dist/<name>.exe` —— PKG 里的 `engine\*` 路径是**反斜杠**，别用正斜杠匹配
2. 启动 exe 数秒后强杀 → 看 `C:\Users\Administrator\AppData\Local\Temp\_MEI*` 临时目录里 `engine/` 释放是否完整

## 用户偏好（项目相关）
- 追求 Apple/极简设计感，审美标准高
- 同一项目有"两个名字"的情况别强行统一，听用户的
- 改动偏好"先简后繁"：图标→8K→通用工具链
- 走过的流程会要求固化为可复用 skill
- **UI 改稿时倾向于说成"PyQt5"——但项目实际是 tkinter**。真要换 PyQt5 会膨胀 onefile 体积 3-4 倍，不值。沿用 tkinter + ttk clam + 调色板可达到同等视觉效果

## Python 数值铁律（多次踩过）
- **`负数**小数 = complex`**：浮点/整数负底数取非整数次方直接变复数（如 `(-0.005) ** 0.55`）。视觉/数学代码里 `sin(...)` 的包络函数必须 `abs(...)` 包一层
- Pillow 12+ 的 `ImageDraw.line/polygon/ellipse` 强制**整数坐标**，`int(round(x))` 统一走 helper

## tkinter 铁律（多次踩过，参见各轮日志）
- **线程安全**：工作线程**绝不能** `self.after(...)` / `createcommand`（`RuntimeError: main thread is not in main loop`）。后台线程只放纯数据进 `queue.Queue`，主线程 `_poll_loop` 消化
- **Canvas.lower() 被同名 item 方法覆盖**：报 TclError "wrong # args"。**修法**：走 widget 层 `canvas.tk.call("lower", canvas._w)`
- **混合 place/grid 同一父**：可用但不优雅，size 报告可能抽风（实测 drop_canvas winfo_width 报 2x）。必要时调 `update_idletasks()` 多轮等布局稳定
- 项目内 `image_upscaler.py` 已有 `_log_queue` + `__PROGRESS__` 事件机制，参照走
