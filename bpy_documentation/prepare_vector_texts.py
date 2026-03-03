#!/usr/bin/env python3
"""
将bpy API文档处理为向量检索所需的文本格式
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List


def process_api_to_text(api: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个API处理为检索文本
    
    Args:
        api: API文档字典
    
    Returns:
        处理后的文档，包含原始信息和检索文本
    """
    # 提取字段，处理缺失值
    description = api.get('description', '').strip()
    name = api.get('name', '').strip()
    page_title = api.get('page_title', '').strip()
    
    # 如果page_title缺失，使用category作为备选
    if not page_title:
        page_title = api.get('category', 'other')
    
    # 如果description缺失，使用full_name
    if not description:
        description = api.get('full_name', name)
    
    # 构建检索文本 - 长文本格式（增加信息量以提高向量区分度）
    full_name = api.get('full_name', '')
    module = api.get('module', '')
    signature = api.get('signature', '')
    category = api.get('category', 'other')
    parameters = api.get('parameters', [])
    
    # 构建参数摘要（包含参数名和简短描述）
    param_parts = []
    for param in parameters[:8]:  # 最多取8个参数
        param_name = param.get('name', '')
        param_type = param.get('type', '').split(',')[0]  # 只取类型的第一部分
        param_desc = param.get('description', '')
        
        # 简化参数描述（取第一句或前50字符）
        if param_desc:
            param_desc_short = param_desc.split('.')[0][:50]
            param_parts.append(f"{param_name} ({param_type}): {param_desc_short}")
        else:
            param_parts.append(f"{param_name} ({param_type})")
    
    param_text = '. '.join(param_parts) if param_parts else 'No parameters'
    
    # 组合文本：函数名 + 描述 + 完整API + 模块 + 参数 + 页面标题
    text_for_embedding = f"{name}: {description}. Full API: {full_name}. Module: {module}. Parameters: {param_text}. Page: {page_title}"
    
    # 构建返回结果
    result = {
        'api_id': api.get('id', api.get('full_name', '')),
        'text_for_embedding': text_for_embedding,
        'metadata': {
            'module': api.get('module', ''),
            'submodule': api.get('submodule', ''),
            'name': name,
            'full_name': api.get('full_name', ''),
            'category': api.get('category', 'other'),
            'type': api.get('type', 'function'),
            'signature': api.get('signature', ''),
            'description': description,
            'page_title': page_title,
            'parameters': api.get('parameters', [])
        }
    }
    
    return result


def process_all_apis(input_file: str, output_file: str) -> None:
    """
    处理所有API文档
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSONL文件路径
    """
    print(f"读取文件: {input_file}")
    
    # 读取原始JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        apis = json.load(f)
    
    total_count = len(apis)
    print(f"总API数量: {total_count}")
    
    # 统计信息
    stats = {
        'total': total_count,
        'no_description': 0,
        'no_page_title': 0,
        'text_lengths': []
    }
    
    # 处理每个API并写入JSONL
    processed_apis = []
    
    print("开始处理...")
    for i, api in enumerate(apis):
        if (i + 1) % 500 == 0:
            print(f"  已处理: {i + 1}/{total_count}")
        
        # 统计缺失字段
        if not api.get('description'):
            stats['no_description'] += 1
        if not api.get('page_title'):
            stats['no_page_title'] += 1
        
        # 处理API
        processed = process_api_to_text(api)
        processed_apis.append(processed)
        
        # 统计文本长度
        stats['text_lengths'].append(len(processed['text_for_embedding']))
    
    # 写入JSONL文件
    print(f"\n保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for processed in processed_apis:
            f.write(json.dumps(processed, ensure_ascii=False) + '\n')
    
    # 计算统计信息
    text_lengths = stats['text_lengths']
    stats['text_length_min'] = min(text_lengths)
    stats['text_length_max'] = max(text_lengths)
    stats['text_length_avg'] = sum(text_lengths) / len(text_lengths)
    stats['text_length_median'] = sorted(text_lengths)[len(text_lengths) // 2]
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理统计:")
    print("=" * 60)
    print(f"总API数: {stats['total']}")
    print(f"缺少description: {stats['no_description']} ({stats['no_description']/stats['total']*100:.1f}%)")
    print(f"缺少page_title: {stats['no_page_title']} ({stats['no_page_title']/stats['total']*100:.1f}%)")
    print(f"\n文本长度统计:")
    print(f"  最短: {stats['text_length_min']} 字符")
    print(f"  最长: {stats['text_length_max']} 字符")
    print(f"  平均: {stats['text_length_avg']:.1f} 字符")
    print(f"  中位数: {stats['text_length_median']} 字符")
    print(f"  预估token数: {stats['text_length_avg']/4:.1f} tokens (按1 token ≈ 4字符估算)")
    print("=" * 60)
    
    # 保存统计信息
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计信息已保存到: {stats_file}")
    
    # 显示几个示例
    print("\n前3个示例:")
    print("-" * 60)
    for i in range(min(3, len(processed_apis))):
        example = processed_apis[i]
        print(f"\n{i+1}. API ID: {example['api_id']}")
        print(f"   文本: {example['text_for_embedding'][:150]}...")
        print(f"   长度: {len(example['text_for_embedding'])} 字符")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    input_file = base_dir / "structured_docs" / "bpy_ops_flat.json"
    output_file = base_dir / "vector_index" / "api_texts.jsonl"
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 处理文件
    process_all_apis(str(input_file), str(output_file))
    
    print("\n完成!")


if __name__ == '__main__':
    main()

