#!/usr/bin/env python3
"""测试单个HTML文件的解析"""

from pathlib import Path
from bs4 import BeautifulSoup
import json

def test_parse_export_anim():
    html_file = Path("/userhome/cs2/u3665834/projects/hunyuan3D-Agent-G1/bpy_documentation/blender_python_reference_4_5/bpy.ops.export_anim.html")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # 找到bpy.ops.export_anim.bvh函数
    func_dl = soup.find('dt', id='bpy.ops.export_anim.bvh')
    if not func_dl:
        print("❌ 未找到 bpy.ops.export_anim.bvh")
        return
    
    print("✓ 找到函数定义")
    func_dl = func_dl.parent  # 获取包含dt和dd的dl元素
    
    # 获取dd
    dd = func_dl.find('dd')
    if not dd:
        print("❌ 未找到dd元素")
        return
    
    print("✓ 找到dd元素")
    
    # 查找field-list
    param_list = dd.find('dl', class_='field-list')
    if not param_list:
        print("❌ 未找到field-list")
        return
    
    print("✓ 找到field-list")
    print(f"  field-list class: {param_list.get('class')}")
    
    # 查找Parameters的dt
    param_dt = None
    all_dts = param_list.find_all('dt')
    print(f"✓ 找到 {len(all_dts)} 个dt元素")
    
    for idx, dt in enumerate(all_dts):
        dt_text = dt.get_text()
        print(f"  dt[{idx}]: {dt_text[:50]}")
        if 'Parameters' in dt_text:
            param_dt = dt
            print(f"  → 这是Parameters!")
    
    if not param_dt:
        print("❌ 未找到Parameters的dt")
        return
    
    print("✓ 找到Parameters的dt")
    
    # 查找对应的dd
    param_dd = param_dt.find_next_sibling('dd')
    if not param_dd:
        print("❌ 未找到Parameters对应的dd")
        return
    
    print("✓ 找到Parameters对应的dd")
    
    # 查找所有li
    param_items = param_dd.find_all('li', recursive=True)
    print(f"✓ 找到 {len(param_items)} 个li元素")
    
    # 解析每个参数
    for idx, li in enumerate(param_items[:3]):  # 只看前3个
        text = li.get_text(strip=True)
        strong = li.find('strong')
        
        if strong:
            param_name = strong.get_text(strip=True)
            remaining = text.replace(param_name, '', 1).strip()
            
            print(f"\n参数 {idx + 1}:")
            print(f"  名称: {param_name}")
            print(f"  剩余文本: {remaining[:100]}")
            
            # 检查分隔符
            if '–' in remaining:
                parts = remaining.split('–', 1)
                print(f"  类型部分: {parts[0][:80]}")
                print(f"  描述部分: {parts[1][:80] if len(parts) > 1 else 'N/A'}")
        else:
            print(f"\n参数 {idx + 1}: 没有<strong>标签")
            print(f"  完整文本: {text[:100]}")

if __name__ == '__main__':
    test_parse_export_anim()

