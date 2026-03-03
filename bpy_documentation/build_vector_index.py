#!/usr/bin/env python3
"""
构建向量索引：对API文档进行向量化并构建FAISS索引
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm
import re


# 同义词扩展（与BM25保持一致）
SYNONYM_DICT = {
    'simulation': 'cache bake ptcache physics',
    'baking': 'cache ptcache bake',
    'caching': 'cache ptcache bake',
    'particle': 'hair fluid smoke',
    'fracture': 'cell fragment',
    'explosion': 'blast burst',
    'fragments': 'pieces shards',
    'trajectories': 'paths motion',
    'shader': 'material node',
    'texture': 'image map',
    'subsurface': 'sss scattering',
    'bump': 'normal displacement',
    'bone': 'armature',
    'skeleton': 'armature bone',
    'ik': 'inverse kinematics',
    'rigging': 'armature bone',
}


def apply_synonyms(text: str) -> str:
    """应用同义词扩展"""
    text_lower = text.lower()
    expanded_terms = []
    
    for original, synonyms in SYNONYM_DICT.items():
        if original in text_lower:
            expanded_terms.append(synonyms)
    
    return ' '.join(expanded_terms)


def enhance_text_for_embedding(text: str, api_id: str) -> str:
    """
    增强文本以提高向量检索质量
    
    策略：
    1. 添加操作类型标签
    2. 添加同义词扩展
    3. 添加API命名空间信息
    4. 为属性添加特殊标签
    """
    enhanced_parts = [text]
    
    # 提取API命名空间信息
    api_lower = api_id.lower()
    text_lower = text.lower()
    
    # 特殊处理：如果是属性（attribute/property），添加属性相关标签
    if '(property)' in text_lower or 'attribute' in api_lower:
        enhanced_parts.append("attribute property value setting configuration parameter")
        
        # 针对特定类型的属性添加更多语义
        if 'energy' in api_lower or 'power' in api_lower:
            enhanced_parts.append("set adjust configure energy power strength intensity")
        
        if 'color' in api_lower or 'colour' in api_lower:
            enhanced_parts.append("set adjust configure color colour tint hue")
        
        if 'influence' in api_lower or 'weight' in api_lower or 'strength' in api_lower:
            enhanced_parts.append("set adjust configure influence weight strength amount factor")
        
        if 'target' in api_lower:
            enhanced_parts.append("set configure target destination object reference")
        
        if 'location' in api_lower or 'position' in api_lower:
            enhanced_parts.append("set configure location position coordinates placement")
        
        if 'rotation' in api_lower or 'orientation' in api_lower:
            enhanced_parts.append("set configure rotation orientation angle euler quaternion")
        
        if 'scale' in api_lower or 'size' in api_lower:
            enhanced_parts.append("set configure scale size dimension magnitude")
    
    # 添加操作类型标签（基于API路径）
    if 'ptcache' in api_lower or 'bake' in api_lower:
        enhanced_parts.append("simulation cache operation for baking and caching data")
    
    if 'rigidbody' in api_lower or 'physics' in api_lower:
        enhanced_parts.append("physics simulation rigid body dynamics operation")
    
    if 'particle' in api_lower or 'hair' in api_lower or 'fluid' in api_lower:
        enhanced_parts.append("particle system hair fluid dynamics operation")
    
    if 'node' in api_lower:
        enhanced_parts.append("node editing shader material operation")
    
    if 'pose' in api_lower or 'armature' in api_lower or 'bone' in api_lower:
        enhanced_parts.append("bone armature rigging animation operation")
    
    if 'constraint' in api_lower:
        enhanced_parts.append("constraint relationship connection operation")
    
    if 'mode_set' in api_lower or 'posemode' in api_lower:
        enhanced_parts.append("mode switching edit pose object operation")
    
    if 'transform' in api_lower:
        enhanced_parts.append("transformation move rotate scale operation")
    
    if 'modifier' in api_lower:
        enhanced_parts.append("modifier geometry modification operation")
    
    if 'sequencer' in api_lower:
        enhanced_parts.append("sequence editor video editing operation")
    
    if 'render' in api_lower:
        enhanced_parts.append("rendering output image generation operation")
    
    if 'keyframe' in api_lower or 'anim' in api_lower:
        enhanced_parts.append("animation keyframe timeline operation")
    
    if 'delete' in api_lower or 'remove' in api_lower or 'clear' in api_lower:
        enhanced_parts.append("deletion removal cleaning operation")
    
    if 'undo' in api_lower or 'redo' in api_lower:
        enhanced_parts.append("undo redo rollback history operation")
    
    # 添加同义词扩展
    synonym_expansion = apply_synonyms(text)
    if synonym_expansion:
        enhanced_parts.append(synonym_expansion)
    
    # 合并并清理
    enhanced_text = ' '.join(enhanced_parts)
    enhanced_text = ' '.join(enhanced_text.split())  # 清理多余空格
    
    return enhanced_text


def process_method_to_text(method: Dict[str, Any], class_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个类方法处理为检索文本（带类信息）
    
    Args:
        method: 方法文档字典
        class_info: 类信息字典
    
    Returns:
        处理后的文档，包含原始信息和检索文本
    """
    # 提取方法字段
    method_name = method.get('name', '').strip()
    method_id = method.get('method_id', '').strip()
    description = method.get('description', '').strip()
    signature = method.get('signature', '')
    parameters = method.get('parameters', [])
    returns = method.get('returns', {})
    is_classmethod = method.get('is_classmethod', False)
    
    # 提取类字段
    class_id = class_info.get('class_id', '')
    class_name = class_info.get('class_name', '')
    page_title = class_info.get('page_title', '')
    class_description = class_info.get('description', '')
    base_class = class_info.get('base_class', '')
    
    # 构建参数摘要
    param_parts = []
    for param in parameters[:8]:  # 最多取8个参数
        param_name = param.get('name', '')
        param_type = param.get('type', '').split(',')[0]  # 只取类型的第一部分
        param_desc = param.get('description', '')
        
        # 简化参数描述
        if param_desc:
            param_desc_short = param_desc.split('.')[0][:50]
            param_parts.append(f"{param_name} ({param_type}): {param_desc_short}")
        else:
            param_parts.append(f"{param_name} ({param_type})")
    
    param_text = '. '.join(param_parts) if param_parts else 'No parameters'
    
    # 构建返回值信息
    return_text = ''
    if returns:
        return_type = returns.get('type', '')
        return_desc = returns.get('description', '')
        if return_type or return_desc:
            return_text = f" Returns: {return_desc} ({return_type})" if return_type else f" Returns: {return_desc}"
    
    # 组合文本：方法名 + 方法描述 + 完整API + 参数 + 返回值 + 类信息
    # 格式与ops保持一致
    text_for_embedding = f"{method_name}: {description}. Full API: {method_id}. Module: {class_id}. Parameters: {param_text}.{return_text} Class: {class_name} ({class_description}). Base: {base_class}. Page: {page_title}"
    
    # 构建返回结果
    result = {
        'api_id': method_id,
        'text_for_embedding': text_for_embedding,
        'metadata': {
            'module': class_id,
            'submodule': 'types',
            'name': method_name,
            'full_name': method_id,
            'category': 'types',
            'type': 'classmethod' if is_classmethod else 'method',
            'signature': signature,
            'description': description,
            'page_title': page_title,
            'parameters': parameters,
            'returns': returns,
            'class_id': class_id,
            'class_name': class_name,
            'class_description': class_description,
            'base_class': base_class
        }
    }
    
    return result


def process_attribute_to_text(attribute: Dict[str, Any], class_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个类属性处理为检索文本（带类信息）
    
    Args:
        attribute: 属性文档字典
        class_info: 类信息字典
    
    Returns:
        处理后的文档，包含原始信息和检索文本
    """
    # 提取属性字段
    attribute_name = attribute.get('name', '').strip()
    attribute_id = attribute.get('attribute_id', '').strip()
    description = attribute.get('description', '').strip()
    attr_type = attribute.get('type', '').strip()
    enum_values = attribute.get('enum_values', None)
    
    # 提取类字段
    class_id = class_info.get('class_id', '')
    class_name = class_info.get('class_name', '')
    page_title = class_info.get('page_title', '')
    class_description = class_info.get('description', '')
    base_class = class_info.get('base_class', '')
    
    # 构建属性类型信息
    type_text = f"Type: {attr_type}" if attr_type else "Type: unspecified"
    
    # 构建枚举值信息（如果存在）
    enum_text = ''
    if enum_values:
        # 如果枚举值是列表，取前5个
        if isinstance(enum_values, list) and enum_values:
            enum_preview = ', '.join(str(e).split()[0] for e in enum_values[:5])
            enum_text = f" Enum values: {enum_preview}"
    
    # 组合文本：属性名 + 属性描述 + 完整API + 类型 + 枚举值（如果有）+ 类信息
    # 格式与method保持一致，但标注为attribute/property
    text_for_embedding = f"{attribute_name} (property): {description}. Full API: {attribute_id}. Module: {class_id}. {type_text}.{enum_text} Class: {class_name} ({class_description}). Base: {base_class}. Page: {page_title}"
    
    # 构建返回结果
    result = {
        'api_id': attribute_id,
        'text_for_embedding': text_for_embedding,
        'metadata': {
            'module': class_id,
            'submodule': 'types',
            'name': attribute_name,
            'full_name': attribute_id,
            'category': 'types',
            'type': 'attribute',
            'attribute_type': attr_type,
            'description': description,
            'page_title': page_title,
            'enum_values': enum_values,
            'class_id': class_id,
            'class_name': class_name,
            'class_description': class_description,
            'base_class': base_class
        }
    }
    
    return result


def load_types_texts(types_file: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    加载bpy.types文档并处理为方法和属性级别的文本（带文本增强）
    
    过滤规则：
    - 跳过 is_classmethod: true 的方法（保留实例方法）
    - 包含所有属性（attributes）
    
    Returns:
        texts: 用于向量化的文本列表（已增强）
        metadata: 对应的元数据列表
    """
    print(f"加载Types文本: {types_file}")
    
    texts = []
    metadata_list = []
    
    with open(types_file, 'r', encoding='utf-8') as f:
        types_data = json.load(f)
    
    total_methods = 0
    total_attributes = 0
    skipped_classmethods = 0
    
    for class_info in tqdm(types_data, desc="处理类"):
        # 处理方法（methods）
        methods = class_info.get('methods', [])
        
        for method in methods:
            # 过滤掉 classmethod
            if method.get('is_classmethod', False):
                skipped_classmethods += 1
                continue
            
            # 处理方法
            processed = process_method_to_text(method, class_info)
            
            # 提取text_for_embedding并增强
            original_text = processed['text_for_embedding']
            api_id = processed['api_id']
            
            # 应用文本增强
            enhanced_text = enhance_text_for_embedding(original_text, api_id)
            
            texts.append(enhanced_text)
            
            # 保存完整metadata用于后续检索
            metadata_list.append({
                'api_id': api_id,
                'metadata': processed['metadata']
            })
            
            total_methods += 1
        
        # 处理属性（attributes）
        attributes = class_info.get('attributes', [])
        
        for attribute in attributes:
            # 处理属性
            processed = process_attribute_to_text(attribute, class_info)
            
            # 提取text_for_embedding并增强
            original_text = processed['text_for_embedding']
            api_id = processed['api_id']
            
            # 应用文本增强
            enhanced_text = enhance_text_for_embedding(original_text, api_id)
            
            texts.append(enhanced_text)
            
            # 保存完整metadata用于后续检索
            metadata_list.append({
                'api_id': api_id,
                'metadata': processed['metadata']
            })
            
            total_attributes += 1
    
    print(f"加载完成: {total_methods} 个实例方法, {total_attributes} 个属性 from {len(types_data)} 个类（已应用文本增强）")
    print(f"已跳过: {skipped_classmethods} 个 classmethods")
    return texts, metadata_list


def load_api_texts(jsonl_file: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    加载API文本数据（ops，带文本增强）
    
    Returns:
        texts: 用于向量化的文本列表（已增强）
        metadata: 对应的元数据列表
    """
    print(f"加载Ops文本: {jsonl_file}")
    
    texts = []
    metadata_list = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # 提取text_for_embedding并增强
            original_text = data['text_for_embedding']
            api_id = data['api_id']
            
            # 应用文本增强
            enhanced_text = enhance_text_for_embedding(original_text, api_id)
            
            texts.append(enhanced_text)
            
            # 保存完整metadata用于后续检索
            metadata_list.append({
                'api_id': api_id,
                'metadata': data['metadata']
            })
    
    print(f"加载完成: {len(texts)} 个API（已应用文本增强）")
    return texts, metadata_list


def download_and_load_model(model_dir: str):
    """
    下载并加载模型
    
    Args:
        model_dir: 模型保存目录
    """
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_path = Path(model_dir)
    
    print(f"\n{'='*60}")
    print("加载sentence-transformers/all-MiniLM-L6-v2模型")
    print(f"{'='*60}")
    
    # 检查是否已下载
    if model_path.exists() and (model_path / "config.json").exists():
        print(f"从本地加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    else:
        print(f"从Hugging Face下载模型到: {model_path}")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 下载并保存
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 保存到本地
        print("保存模型到本地...")
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        print("模型保存完成")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载到设备: {device}")
    print(f"模型配置: {model.config}")
    
    return tokenizer, model, device


def encode_texts_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """
    批量编码文本为向量
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        model: 模型
        device: 设备
        batch_size: 批次大小
    
    Returns:
        embeddings: numpy数组 shape=(len(texts), embedding_dim)
    """
    import torch
    
    print(f"\n开始向量化 {len(texts)} 个文本 (批次大小: {batch_size})")
    
    all_embeddings = []
    
    # 批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="向量化进度"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenization
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 编码
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 获取embeddings - 通常是[CLS] token的输出或pooler_output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # 使用最后一层的[CLS] token (第一个token)
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            # L2 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # 转为numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
    
    # 合并所有批次
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"向量化完成: shape={all_embeddings.shape}")
    return all_embeddings


def build_faiss_index(embeddings: np.ndarray, index_type: str = "IndexFlatIP"):
    """
    构建FAISS索引
    
    Args:
        embeddings: 向量数组
        index_type: 索引类型
    
    Returns:
        index: FAISS索引
    """
    import faiss
    
    print(f"\n构建FAISS索引 (类型: {index_type})")
    
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    
    print(f"向量维度: {dimension}")
    print(f"向量数量: {num_vectors}")
    
    if index_type == "IndexFlatIP":
        # 内积索引（用于归一化向量的余弦相似度）
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexHNSWFlat":
        # HNSW图索引
        index = faiss.IndexHNSWFlat(dimension, 32)
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")
    
    # 添加向量
    print("添加向量到索引...")
    index.add(embeddings.astype('float32'))
    
    print(f"索引构建完成: {index.ntotal} 个向量")
    return index


def save_index_and_metadata(
    index,
    embeddings: np.ndarray,
    metadata_list: List[Dict],
    output_dir: str,
    model_info: Dict[str, Any],
    texts: List[str] = None
):
    """
    保存索引和元数据
    
    Args:
        texts: 可选，embedding前的原始文本，用于查看样本
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存索引到: {output_dir}")
    
    # 保存FAISS索引
    import faiss
    index_file = output_path / "faiss_index.bin"
    faiss.write_index(index, str(index_file))
    print(f"FAISS索引已保存: {index_file}")
    
    # 保存embeddings (numpy格式)
    embeddings_file = output_path / "embeddings.npy"
    np.save(str(embeddings_file), embeddings)
    print(f"Embeddings已保存: {embeddings_file}")
    
    # 保存metadata
    metadata_file = output_path / "api_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_docs': len(metadata_list),
            'embedding_dim': embeddings.shape[1],
            'model_name': model_info['model_name'],
            'index_type': model_info['index_type'],
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_mappings': metadata_list
        }, f, ensure_ascii=False, indent=2)
    print(f"Metadata已保存: {metadata_file}")
    
    # 保存配置
    config_file = output_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_info['model_name'],
            'embedding_dim': embeddings.shape[1],
            'index_type': model_info['index_type'],
            'num_docs': len(metadata_list),
            'preprocessing': {
                'replace_pipe': True,
                'clean_whitespace': True,
                'normalize': True
            },
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, ensure_ascii=False, indent=2)
    print(f"配置已保存: {config_file}")
    
    # 保存原始文本样本（embedding前的文本，用于直观查看）
    if texts:
        samples_file = output_path / "embedding_text_samples.json"
        
        # 保存前100个样本（ops和types的混合）
        samples = []
        
        # 前50个（主要是ops）
        for i in range(min(50, len(texts))):
            samples.append({
                'index': i,
                'api_id': metadata_list[i]['api_id'],
                'category': metadata_list[i]['metadata'].get('category', 'unknown'),
                'text': texts[i]
            })
        
        # 后50个（主要是types）
        if len(texts) > 50:
            start_idx = max(50, len(texts) - 50)
            for i in range(start_idx, len(texts)):
                samples.append({
                    'index': i,
                    'api_id': metadata_list[i]['api_id'],
                    'category': metadata_list[i]['metadata'].get('category', 'unknown'),
                    'text': texts[i]
                })
        
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_texts': len(texts),
                'sample_count': len(samples),
                'note': 'These are the texts BEFORE embedding (with enhancements applied)',
                'samples': samples
            }, f, ensure_ascii=False, indent=2)
        print(f"文本样本已保存: {samples_file} ({len(samples)} 个样本)")
    
    print(f"\n所有文件已保存到: {output_dir}")


def test_search(index, texts: List[str], metadata_list: List[Dict], tokenizer, model, device):
    """
    测试检索功能
    """
    print(f"\n{'='*60}")
    print("测试检索功能")
    print(f"{'='*60}")
    
    test_queries = [
        "add camera to scene",
        "create cube mesh",
        "render image",
        "export glb file",
        "set material color"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 向量化查询
        query_processed = query.replace(' | ', ' ')
        query_processed = ' '.join(query_processed.split())
        
        import torch
        inputs = tokenizer([query_processed], padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                query_embedding = outputs.pooler_output
            else:
                query_embedding = outputs.last_hidden_state[:, 0, :]
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            query_embedding = query_embedding.cpu().numpy()
        
        # 搜索
        k = 5
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        print(f"Top-{k} 结果:")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            api_id = metadata_list[idx]['api_id']
            text = texts[idx][:80]
            print(f"  {rank}. [{score:.3f}] {api_id}")
            print(f"     {text}...")


def main():
    """主函数"""
    # 配置路径
    base_dir = Path(__file__).parent
    ops_file = base_dir / "vector_index" / "api_texts.jsonl"
    types_file = base_dir / "structured_docs" / "bpy_types_flat.json"
    output_dir = base_dir / "vector_index"
    model_dir = base_dir / "all-MiniLM-L6-v2"
    
    
    # 检查输入文件
    if not ops_file.exists():
        print(f"错误: Ops文件不存在: {ops_file}")
        return
    
    if not types_file.exists():
        print(f"错误: Types文件不存在: {types_file}")
        return
    
    print(f"{'='*60}")
    print("BPY API 向量索引构建 - 包含Ops和Types")
    print(f"{'='*60}")
    print(f"Ops文件: {ops_file}")
    print(f"Types文件: {types_file}")
    print(f"输出目录: {output_dir}")
    print(f"模型目录: {model_dir}")
    print(f"{'='*60}\n")
    
    # 1. 加载Ops文本数据
    print("\n" + "="*60)
    print("步骤 1/5: 加载Ops文本")
    print("="*60)
    ops_texts, ops_metadata = load_api_texts(str(ops_file))
    
    # 2. 加载Types文本数据（方法级别）
    print("\n" + "="*60)
    print("步骤 2/5: 加载Types文本（方法级别）")
    print("="*60)
    types_texts, types_metadata = load_types_texts(str(types_file))
    
    # 3. 合并Ops和Types
    print("\n" + "="*60)
    print("步骤 3/5: 合并Ops和Types")
    print("="*60)
    all_texts = ops_texts + types_texts
    all_metadata = ops_metadata + types_metadata
    
    print(f"总文档数: {len(all_texts)}")
    print(f"  - Ops: {len(ops_texts)}")
    print(f"  - Types方法: {len(types_texts)}")
    
    # 4. 加载模型
    print("\n" + "="*60)
    print("步骤 4/5: 加载模型")
    print("="*60)
    tokenizer, model, device = download_and_load_model(str(model_dir))
    
    # 5. 向量化
    print("\n" + "="*60)
    print("步骤 5/5: 向量化")
    print("="*60)
    start_time = time.time()
    embeddings = encode_texts_batch(
        all_texts,
        tokenizer,
        model,
        device,
        batch_size=32  # 根据GPU内存调整
    )
    encoding_time = time.time() - start_time
    print(f"\n向量化耗时: {encoding_time:.2f} 秒")
    print(f"平均速度: {len(all_texts)/encoding_time:.1f} docs/sec")
    
    # 6. 构建FAISS索引
    index = build_faiss_index(embeddings, index_type="IndexFlatIP")
    
    # 7. 保存（包括原始文本样本）
    model_info = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'index_type': 'IndexFlatIP'
    }
    save_index_and_metadata(index, embeddings, all_metadata, str(output_dir), model_info, texts=all_texts)
    
    # 8. 测试检索
    test_search(index, all_texts, all_metadata, tokenizer, model, device)
    
    print(f"\n{'='*60}")
    print("构建完成!")
    print(f"说明：")
    print(f"- 已合并Ops ({len(ops_texts)}) 和 Types方法 ({len(types_texts)})")
    print(f"- 总向量数: {len(all_texts)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

