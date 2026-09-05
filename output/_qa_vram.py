"""质检: 级联 Vulkan + 显存监控 + 后处理"""
import sys, time, shutil, subprocess
from pathlib import Path
sys.path.insert(0, r'E:\xiaomubiao\pyt')
import image_upscaler as M
from PIL import Image
import ast
ast.parse(open(r'E:\xiaomubiao\pyt\image_upscaler.py', 'r', encoding='utf-8').read())
print('语法 OK')

tmp = Path(r'E:\xiaomubiao\pyt\output\_qa8')
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)

# 模拟你那张照片（247x300）
img = Image.new('RGB', (247, 300), (160, 100, 50))
for y in range(50, 250, 5):
    for x in range(80, 220, 5):
        v = (y*7 + x*3) % 255
        for dy in range(2):
            for dx in range(2):
                if y+dy < 300 and x+dx < 247:
                    img.putpixel((x+dx, y+dy), (v, 200-v//2, 100))
img.save(tmp / 'src.png')

engine = M.find_engine_path()
print('engine =', engine)
print('SCALE_OPTIONS =', M.SCALE_OPTIONS)
print('VULKAN_TILE_PIXEL_LIMIT =', M.VULKAN_TILE_PIXEL_LIMIT)
print('VULKAN_TILE_SIZE =', M.VULKAN_TILE_SIZE)
print()

# 启动显存监控线程
import threading
peak_vram = [0]
stop = [False]
def monitor():
    while not stop[0]:
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                used, total = r.stdout.strip().split(',')
                used_mb = int(used.strip())
                if used_mb > peak_vram[0]:
                    peak_vram[0] = used_mb
        except Exception:
            pass
        time.sleep(0.2)
t = threading.Thread(target=monitor, daemon=True)
t.start()

# 主任务
dst = tmp / 'out_4k.png'
t0 = time.time()
ok, err, used = M._target_preset_dispatch(engine, None, tmp/'src.png', dst, '4K', 'realesrgan-x4plus')
dt = time.time() - t0
stop[0] = True
time.sleep(0.3)

print(f'ok={ok}  engine={used}  耗时={dt:.2f}s')
if dst.exists():
    im = Image.open(dst)
    print(f'  输出尺寸 = {im.size}  文件 = {dst.stat().st_size/1024:.1f}KB')
print(f'  显存峰值 = {peak_vram[0]} MB  / 6144 MB  ({peak_vram[0]/6144*100:.1f}%)')
if err: print(f'  err={err[:200]}')

shutil.rmtree(tmp, ignore_errors=True)
print()
print('=== 质检: 全部完成 ===')
