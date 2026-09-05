"""质检脚本: 验证 4K/2K/FHD 流水线"""
import sys, time, shutil
from pathlib import Path
sys.path.insert(0, r'E:\xiaomubiao\pyt')
import image_upscaler as M
from PIL import Image

tmp = Path(r'E:\xiaomubiao\pyt\output\_qa6')
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)

cases = [
    ('A. 247x300 -> 4K',           (247, 300),  '4K'),
    ('B. 800x600 -> 2K',           (800, 600),  '2K'),
    ('C. 4000x3000 -> FHD(shink)', (4000, 3000), 'FHD'),
    ('D. 1100x800 -> 4K',          (1100, 800), '4K'),
    ('E. 600x400 -> FHD(sharp)',   (600, 400),  'FHD'),
]

engine = M.find_engine_path()
print('engine =', engine)
print('SCALE_OPTIONS =', M.SCALE_OPTIONS)
print('TARGET_PRESETS =', M.TARGET_PRESETS)
print()

for name, sz, preset in cases:
    src = tmp / ('src_%dx%d.png' % sz)
    img = Image.new('RGB', sz, (180, 100, 60))
    img.save(src)
    dst = tmp / ('out_%s.png' % preset)
    t0 = time.time()
    ok, err, used = M._target_preset_dispatch(engine, None, src, dst, preset, 'realesrgan-x4plus')
    dt = time.time() - t0
    if dst.exists():
        out_size = Image.open(dst).size
    else:
        out_size = 'NONE'
    expect_long = M.TARGET_PRESETS[preset]
    if isinstance(out_size, tuple):
        actual_long = max(out_size)
        passed = abs(actual_long - expect_long) <= 1
    else:
        actual_long = 0
        passed = False
    print(name)
    print('  -> size=%s engine=%s dt=%.2fs expect=%d %s' % (
        out_size, used, dt, expect_long, 'PASS' if passed else 'FAIL'
    ))

print()
print('F. 异常: 文件不存在')
ok, err, used = M._target_preset_dispatch(engine, None, tmp/'no_such.png', tmp/'fail.png', '4K', 'realesrgan-x4plus')
print('  ok=%s err=%s' % (ok, err[:80]))

print('G. 异常: 无效档位')
src_a = tmp / 'src_247x300.png'
ok, err, used = M._target_preset_dispatch(engine, None, src_a, tmp/'fail2.png', 'INVALID', 'realesrgan-x4plus')
print('  ok=%s err=%s' % (ok, err[:80]))

shutil.rmtree(tmp, ignore_errors=True)
print()
print('=== 质检第 3 遍: 全部完成 ===')
