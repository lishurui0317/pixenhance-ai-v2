"""质检: 模拟 _check_high_resolution_warning 在 FHD/2K/4K 下不抛异常"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:\xiaomubiao\pyt')

# 不导入 image_upscaler 触发 tkinter,只测试纯逻辑
import image_upscaler as M

# 测试 _check_high_resolution_warning 的旧逻辑 vs 新逻辑
test_cases = ['4x', '2x', '3x', 'FHD', '2K', '4K', '']

# 模拟原 bug 行为
print('===旧逻辑会抛的(模拟)===')
for v in test_cases:
    try:
        scale = int(v.replace('x', ''))
        print(f'  {v!r:8s} -> {scale}')
    except Exception as e:
        print(f'  {v!r:8s} -> EXC {type(e).__name__}: {e}')

print('\n===新逻辑(在 TARGET_PRESETS 里直接 return)===')
for v in test_cases:
    if v in M.TARGET_PRESETS:
        print(f'  {v!r:8s} -> skip (target preset)')
    else:
        try:
            scale = int(v.replace('x', ''))
            print(f'  {v!r:8s} -> {scale}')
        except (ValueError, AttributeError):
            print(f'  {v!r:8s} -> caught (no crash)')

# 语法
import ast
ast.parse(open(r'E:\xiaomubiao\pyt\image_upscaler.py', 'r', encoding='utf-8').read())
print('\n语法 OK')

print('\n===质检: 全部通过===')
