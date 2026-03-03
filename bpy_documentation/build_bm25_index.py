#!/usr/bin/env python3
"""
构建BM25索引用于API文档检索
优化版：包含同义词词典和查询扩展
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import spacy


# 全局变量
nlp = None

# 同义词词典（扩展版）
SYNONYM_DICT = {
    # 术语映射
    'lamp': 'light',
    'icosphere': 'ico sphere',
    'icosahedral': 'ico',
    
    # 动词同义词 - 变换操作
    'translate': 'move',
    'shift': 'translate move',
    'turn': 'rotate',
    'spin': 'rotate',
    'stretch': 'resize scale',
    
    # 格式变体
    'glb': 'gltf',
    'gltf2': 'gltf',
    
    # Blender特定术语
    'empties': 'empty',
    'armatures': 'armature',
    
    # 仿真相关（Case3的核心问题）
    'simulation': 'cache bake ptcache physics',
    'baking': 'cache ptcache bake',
    'caching': 'cache ptcache bake',
    
    # 粒子和特效
    'particle': 'hair fluid smoke',
    'fracture': 'cell fragment',
    'explosion': 'blast burst',
    'fragments': 'pieces shards',
    'trajectories': 'paths motion',
    'debris': 'fragments pieces',
    
    # 节点相关（Case2的问题）
    'shader': 'material node',
    'texture': 'image map',
    'subsurface': 'sss scattering',
    'bump': 'normal displacement',
    
    # 模式切换
    'edit': 'mode',
    'pose': 'mode',
    
    # 骨骼和动画
    'bone': 'armature',
    'skeleton': 'armature bone',
    'ik': 'inverse kinematics',
    'rigging': 'armature bone',
    
    # 修改器
    'subsurf': 'subdivision',
    'subdivision': 'subsurf',
    'bevel': 'chamfer',
    'solidify': 'thickness',
    
    # 渲染
    'render': 'rendering output',
    'output': 'export save',
    
    # 清理/删除操作
    'clean': 'delete remove clear',
    'clear': 'delete remove',
    'remove': 'delete',
    'rollback': 'undo',
}

# API命名模式映射（扩展版）
API_PATTERNS = {
    # 基础对象
    'camera': ['camera_add', 'bpy.ops.object.camera'],
    'light': ['light_add', 'bpy.ops.object.light', 'lamp'],
    'material': ['material_slot_add', 'bpy.ops.object.material', 'bpy.ops.material'],
    'mesh': ['bpy.ops.mesh', 'primitive'],
    'curve': ['bpy.ops.curve', 'bezier', 'nurbs'],
    
    # 修改器和约束
    'modifier': ['modifier_add', 'bpy.ops.object.modifier'],
    'constraint': ['constraint_add', 'bpy.ops.object.constraint', 'bpy.ops.pose.constraint'],
    
    # 动画
    'keyframe': ['keyframe_insert', 'bpy.ops.anim.keyframe'],
    'animation': ['anim', 'keyframe'],
    
    # 仿真和缓存（Case3关键）
    'simulation': ['ptcache', 'bake', 'rigidbody', 'physics', 'fluid', 'smoke'],
    'cache': ['ptcache', 'bake', 'free_bake'],
    'bake': ['ptcache.bake', 'bake', 'cache'],
    'physics': ['rigidbody', 'physics', 'collision'],
    
    # 粒子系统
    'particle': ['object.particle', 'particle_system', 'hair'],
    
    # 节点操作（Case2关键）
    'node': ['node.add_node', 'node.link', 'shader'],
    'shader': ['node', 'material', 'bpy.ops.node'],
    'connect': ['link', 'node.link'],
    'link': ['node.link', 'connect'],
    
    # 变换操作
    'transform': ['translate', 'rotate', 'resize', 'scale', 'bpy.ops.transform'],
    'move': ['translate', 'bpy.ops.transform.translate'],
    'rotate': ['bpy.ops.transform.rotate'],
    'scale': ['resize', 'bpy.ops.transform.resize'],
    
    # 模式切换
    'mode': ['mode_set', 'bpy.ops.object.mode_set', 'posemode_toggle'],
    'edit': ['mode_set', 'editmode'],
    'pose': ['posemode', 'pose.constraint'],
    
    # 骨骼系统
    'armature': ['armature_add', 'bone', 'bpy.ops.object.armature'],
    'bone': ['armature', 'bone_primitive_add', 'bpy.ops.armature'],
    'ik': ['inverse_kinematics', 'constraint', 'pose.constraint'],
    
    # 渲染
    'render': ['bpy.ops.render', 'rendering'],
    'output': ['save', 'export', 'file'],
    
    # 文件操作
    'save': ['wm.save', 'file', 'export'],
    'open': ['wm.open', 'file', 'import'],
    'export': ['export', 'save'],
    'import': ['import', 'open'],
    
    # 场景管理
    'scene': ['bpy.ops.scene', 'collection'],
    'collection': ['bpy.ops.collection', 'group'],
    
    # 删除/清理
    'delete': ['bpy.ops.object.delete', 'remove', 'clear'],
    'clean': ['delete', 'remove', 'clear', 'free'],
    'clear': ['delete', 'remove', 'free'],
    
    # 序列编辑器（Case3）
    'sequencer': ['bpy.ops.sequencer', 'sequence'],
    
    # 撤销/重做
    'undo': ['bpy.ops.ed.undo', 'rollback'],
    'rollback': ['undo', 'bpy.ops.ed.undo'],
}


def apply_synonyms(text: str) -> str:
    """
    应用同义词替换
    """
    text_lower = text.lower()
    
    for original, synonym in SYNONYM_DICT.items():
        # 使用词边界匹配，避免部分匹配
        pattern = r'\b' + re.escape(original) + r'\b'
        text_lower = re.sub(pattern, synonym, text_lower)
    
    return text_lower


def expand_query(query: str) -> str:
    """
    扩展查询：添加API命名模式和相关术语
    """
    query_lower = query.lower()
    expanded_terms = []
    
    # 添加原始查询
    expanded_terms.append(query_lower)
    
    # 根据关键词添加API模式
    for keyword, patterns in API_PATTERNS.items():
        if keyword in query_lower:
            expanded_terms.extend(patterns)
    
    # 动词扩展模式
    if 'add' in query_lower or 'create' in query_lower:
        # 提取可能的对象名词
        words = query_lower.split()
        for word in words:
            if word not in ['add', 'create', 'the', 'to', 'a', 'an', 'in', 'on']:
                # 添加 xxx_add 和 add_xxx 模式
                expanded_terms.append(f"{word}_add")
                expanded_terms.append(f"add_{word}")
    
    if 'set' in query_lower:
        # 提取可能的对象名词
        words = query_lower.split()
        for word in words:
            if word not in ['set', 'the', 'to', 'a', 'an', 'in', 'on']:
                # 添加 xxx_set 和 set_xxx 模式
                expanded_terms.append(f"{word}_set")
                expanded_terms.append(f"set_{word}")
    
    if 'insert' in query_lower:
        # 提取可能的对象名词
        words = query_lower.split()
        for word in words:
            if word not in ['insert', 'the', 'to', 'a', 'an', 'in', 'on']:
                # 添加 xxx_insert 和 insert_xxx 模式
                expanded_terms.append(f"{word}_insert")
                expanded_terms.append(f"insert_{word}")
    
    if 'utils' in query_lower:
        # 添加 utils 相关模式
        words = query_lower.split()
        for word in words:
            if word not in ['utils', 'the', 'a', 'an', 'in', 'on']:
                # 添加 xxx_utils 模式
                expanded_terms.append(f"{word}_utils")
    
    return ' '.join(expanded_terms)


def load_spacy_model():
    """加载spaCy模型"""
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
        print("已加载spaCy英文模型")
    except OSError:
        print("警告: 未找到spaCy英文模型，将使用基础分词")
        nlp = None


def extract_api_paths(text: str) -> List[str]:
    """
    提取并保留完整的API路径
    例如: bpy.ops.object.camera_add, bpy.data.objects
    """
    api_paths = []
    
    # 匹配bpy.开头的完整路径
    bpy_patterns = re.findall(r'\bbpy\.[a-zA-Z0-9_.]+', text)
    api_paths.extend(bpy_patterns)
    
    # 匹配其他API模式
    api_patterns = re.findall(r'\b[a-zA-Z_]+_[a-zA-Z_]+', text)  # 如camera_add
    api_paths.extend(api_patterns)
    
    return api_paths


def tokenize_text(text: str, apply_synonym: bool = True) -> List[str]:
    """
    针对API文档的优化分词策略
    
    策略：
    1. 应用同义词替换
    2. 保留API名称的完整性（如camera_add, bpy.ops.object.camera_add）
    3. 使用spaCy进行专业分词
    4. 拆分点分隔的路径（bpy.ops.object → bpy, ops, object）
    5. 转小写以提高匹配率
    """
    # 应用同义词
    if apply_synonym:
        text = apply_synonyms(text)
    
    tokens = []
    
    # 1. 首先提取并保留完整的API路径
    api_paths = extract_api_paths(text)
    for api_path in api_paths:
        tokens.append(api_path.lower())
        
        # 同时拆分点分隔的路径
        if '.' in api_path:
            path_parts = api_path.split('.')
            tokens.extend([part.lower() for part in path_parts if len(part) > 1])
    
    # 2. 使用spaCy进行专业分词（如果可用）
    if nlp is not None:
        doc = nlp(text)
        for token in doc:
            if not token.is_space and not token.is_punct and len(token.text) > 1:
                tokens.append(token.lemma_.lower())
    else:
        # 回退到基础分词
        words = re.findall(r'\b\w+\b', text.lower())
        tokens.extend(words)
    
    # 3. 处理下划线分隔的词
    additional_tokens = []
    for token in tokens:
        if '_' in token and '.' not in token:  # 避免重复处理API路径
            sub_words = token.split('_')
            additional_tokens.extend([word for word in sub_words if len(word) > 1])
    
    tokens.extend(additional_tokens)
    
    # 4. 处理驼峰命名
    camel_tokens = []
    for token in tokens:
        if '.' not in token:  # 避免处理API路径
            camel_parts = re.findall(r'[a-z]+|[A-Z][a-z]*', token)
            if len(camel_parts) > 1:
                camel_tokens.extend([p.lower() for p in camel_parts if len(p) > 1])
    
    tokens.extend(camel_tokens)
    
    # 去重并保持顺序
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen and len(token) > 1:  # 过滤单字符
            seen.add(token)
            unique_tokens.append(token)
    
    return unique_tokens


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


def load_types_documents(types_file: str) -> tuple[List[str], List[List[str]], List[Dict]]:
    """
    加载bpy.types文档并处理为方法和属性级别的chunk
    
    过滤规则：
    - 跳过 is_classmethod: true 的方法（保留实例方法）
    - 包含所有属性（attributes）
    
    Returns:
        texts: 用于展示的原始文本
        tokenized_docs: 分词后的文档（用于BM25）
        metadata: 元数据
    """
    print(f"加载Types文档: {types_file}")
    
    texts = []
    tokenized_docs = []
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
            
            # 原始文本
            text = processed['text_for_embedding']
            texts.append(text)
            
            # 分词
            tokens = tokenize_text(text)
            tokenized_docs.append(tokens)
            
            # 元数据
            metadata_list.append({
                'api_id': processed['api_id'],
                'metadata': processed['metadata']
            })
            
            total_methods += 1
        
        # 处理属性（attributes）
        attributes = class_info.get('attributes', [])
        
        for attribute in attributes:
            # 处理属性
            processed = process_attribute_to_text(attribute, class_info)
            
            # 原始文本
            text = processed['text_for_embedding']
            texts.append(text)
            
            # 分词
            tokens = tokenize_text(text)
            tokenized_docs.append(tokens)
            
            # 元数据
            metadata_list.append({
                'api_id': processed['api_id'],
                'metadata': processed['metadata']
            })
            
            total_attributes += 1
    
    print(f"加载完成: {total_methods} 个实例方法, {total_attributes} 个属性 from {len(types_data)} 个类")
    print(f"已跳过: {skipped_classmethods} 个 classmethods")
    return texts, tokenized_docs, metadata_list


def load_api_documents(jsonl_file: str) -> tuple[List[str], List[str], List[Dict]]:
    """
    加载API文档（ops）
    
    Returns:
        texts: 用于展示的原始文本
        tokenized_docs: 分词后的文档（用于BM25）
        metadata: 元数据
    """
    print(f"加载Ops文档: {jsonl_file}")
    
    texts = []
    tokenized_docs = []
    metadata_list = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载文档"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # 原始文本
            text = data['text_for_embedding']
            texts.append(text)
            
            # 分词
            tokens = tokenize_text(text)
            tokenized_docs.append(tokens)
            
            # 元数据
            metadata_list.append({
                'api_id': data['api_id'],
                'metadata': data['metadata']
            })
    
    print(f"加载完成: {len(texts)} 个文档")
    return texts, tokenized_docs, metadata_list


def build_bm25_index(tokenized_docs: List[List[str]], k1: float = 1.2, b: float = 0.75) -> BM25Okapi:
    """
    构建BM25索引，显式设置参数
    
    Args:
        tokenized_docs: 分词后的文档列表
        k1: 词频饱和度参数，控制词频对分数的影响
        b: 长度归一化参数，控制文档长度对分数的影响
    
    Returns:
        BM25索引对象
    """
    print(f"\n构建BM25索引 (k1={k1}, b={b})...")
    
    # 计算平均文档长度
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    
    # 创建BM25索引，显式设置参数
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    
    print(f"索引构建完成: {len(tokenized_docs)} 个文档")
    print(f"BM25参数: k1={k1}, b={b}")
    
    # 统计词汇表信息
    vocab = set()
    for doc in tokenized_docs:
        vocab.update(doc)
    
    print(f"词汇表大小: {len(vocab)} 个唯一词")
    
    # 统计文档长度
    print(f"平均文档长度: {avg_doc_length:.1f} tokens")
    print(f"最短文档: {min(doc_lengths)} tokens")
    print(f"最长文档: {max(doc_lengths)} tokens")
    
    # 针对API文档的参数建议
    if avg_doc_length < 50:
        print("建议: 检测到短文档，可考虑降低k1值以提高精确匹配的重要性")
    elif avg_doc_length > 200:
        print("建议: 检测到长文档，可考虑增加b值以更好地归一化文档长度")
    
    return bm25


def save_bm25_index(
    bm25: BM25Okapi,
    texts: List[str],
    tokenized_docs: List[List[str]],
    metadata_list: List[Dict],
    output_dir: str
):
    """
    保存BM25索引和相关数据
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存索引到: {output_dir}")
    
    # 保存BM25对象
    bm25_file = output_path / "bm25_index.pkl"
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25索引已保存: {bm25_file}")
    
    # 保存分词后的文档（用于重建索引）
    tokenized_file = output_path / "tokenized_docs.pkl"
    with open(tokenized_file, 'wb') as f:
        pickle.dump(tokenized_docs, f)
    print(f"分词文档已保存: {tokenized_file}")
    
    # 保存原始文本
    texts_file = output_path / "bm25_texts.json"
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"原始文本已保存: {texts_file}")
    
    # 保存元数据
    metadata_file = output_path / "bm25_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_docs': len(texts),
            'index_type': 'BM25Okapi',
            'tokenizer': 'custom_api_tokenizer_with_synonyms',
            'api_mappings': metadata_list
        }, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {metadata_file}")
    
    # 保存同义词词典和查询扩展配置
    config_file = output_path / "bm25_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'synonyms': SYNONYM_DICT,
            'api_patterns': API_PATTERNS,
            'query_expansion_enabled': True,
            'bm25_parameters': {
                'k1': bm25.k1,
                'b': bm25.b
            },
            'tokenizer': 'spacy_enhanced_with_api_paths'
        }, f, ensure_ascii=False, indent=2)
    print(f"配置文件已保存: {config_file}")


def test_bm25_search(
    bm25: BM25Okapi,
    texts: List[str],
    metadata_list: List[Dict],
    top_k: int = 5,
    use_query_expansion: bool = True
):
    """
    测试BM25检索功能（优化版）
    """
    print(f"\n{'='*60}")
    print(f"测试BM25检索功能 (查询扩展: {'开启' if use_query_expansion else '关闭'})")
    print(f"{'='*60}")
    
    test_queries = [
        "add camera to scene",
        "create cube mesh", 
        "render image",
        "export glb file",
        "set material color",
        "add lamp object",
        "create icosphere"
    ]
    
    for query in test_queries:
        print(f"\n原始查询: {query}")
        
        # 查询扩展
        if use_query_expansion:
            expanded_query = expand_query(query)
            print(f"扩展查询: {expanded_query[:100]}...")
            query_to_use = expanded_query
        else:
            query_to_use = query
        
        # 分词查询
        query_tokens = tokenize_text(query_to_use)
        print(f"查询分词 ({len(query_tokens)}个): {query_tokens[:15]}{'...' if len(query_tokens) > 15 else ''}")
        
        # BM25检索
        scores = bm25.get_scores(query_tokens)
        
        # 获取Top-K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        print(f"Top-{top_k} 结果:")
        for rank, idx in enumerate(top_indices, 1):
            api_id = metadata_list[idx]['api_id']
            score = scores[idx]
            print(f"  {rank}. [分数:{score:.2f}] {api_id}")


def analyze_token_distribution(tokenized_docs: List[List[str]]):
    """
    分析词频分布
    """
    from collections import Counter
    
    print(f"\n{'='*60}")
    print("词频统计")
    print(f"{'='*60}")
    
    # 统计所有词
    all_tokens = []
    for doc in tokenized_docs:
        all_tokens.extend(doc)
    
    counter = Counter(all_tokens)
    
    print(f"\n最常见的20个词:")
    for token, count in counter.most_common(20):
        print(f"  {token}: {count}")
    
    # 统计API相关的关键词
    api_keywords = ['add', 'remove', 'delete', 'create', 'set', 'get', 
                    'render', 'export', 'import', 'camera', 'mesh', 'material']
    
    print(f"\n关键API动词/名词的频率:")
    for keyword in api_keywords:
        count = counter.get(keyword, 0)
        if count > 0:
            print(f"  {keyword}: {count}")


def main():
    """主函数"""
    # 配置路径
    base_dir = Path(__file__).parent
    ops_file = base_dir / "vector_index" / "api_texts.jsonl"
    types_file = base_dir / "structured_docs" / "bpy_types_flat.json"
    output_dir = base_dir / "bm25_index"
    
    # 检查输入文件
    if not ops_file.exists():
        print(f"错误: Ops文件不存在: {ops_file}")
        return
    
    if not types_file.exists():
        print(f"错误: Types文件不存在: {types_file}")
        return
    
    print(f"{'='*60}")
    print("BPY API BM25索引构建 - 包含Ops和Types")
    print(f"{'='*60}")
    print(f"Ops文件: {ops_file}")
    print(f"Types文件: {types_file}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    # 1. 加载spaCy模型
    load_spacy_model()
    
    # 2. 加载Ops文档
    print("\n" + "="*60)
    print("步骤 1/3: 加载Ops文档")
    print("="*60)
    ops_texts, ops_tokenized, ops_metadata = load_api_documents(str(ops_file))
    
    # 3. 加载Types文档（方法级别）
    print("\n" + "="*60)
    print("步骤 2/3: 加载Types文档（方法级别）")
    print("="*60)
    types_texts, types_tokenized, types_metadata = load_types_documents(str(types_file))
    
    # 4. 合并Ops和Types
    print("\n" + "="*60)
    print("步骤 3/3: 合并Ops和Types")
    print("="*60)
    all_texts = ops_texts + types_texts
    all_tokenized = ops_tokenized + types_tokenized
    all_metadata = ops_metadata + types_metadata
    
    print(f"总文档数: {len(all_texts)}")
    print(f"  - Ops: {len(ops_texts)}")
    print(f"  - Types方法: {len(types_texts)}")
    
    # 5. 分析词频分布
    analyze_token_distribution(all_tokenized)
    
    # 6. 构建BM25索引（针对API文档优化的参数）
    # k1=1.0: 降低词频饱和度，适合短文档
    # b=0.6: 降低长度归一化，因为API文档长度差异较大
    bm25 = build_bm25_index(all_tokenized, k1=0.5, b=0.6)
    
    # 7. 保存索引
    save_bm25_index(bm25, all_texts, all_tokenized, all_metadata, str(output_dir))
    
    # 8. 测试检索 - 对比优化前后
    print(f"\n{'#'*60}")
    print("# 对比测试：优化前 vs 优化后")
    print(f"{'#'*60}")
    
    # 测试1：不使用查询扩展（基线）
    test_bm25_search(bm25, all_texts, all_metadata, top_k=5, use_query_expansion=False)
    
    # 测试2：使用查询扩展（优化版）
    test_bm25_search(bm25, all_texts, all_metadata, top_k=5, use_query_expansion=True)
    
    print(f"\n{'='*60}")
    print("构建完成!")
    print(f"说明：")
    print(f"- 已合并Ops ({len(ops_texts)}) 和 Types方法 ({len(types_texts)})")
    print(f"- 已使用spaCy专业分词器")
    print(f"- 已保留完整API路径")
    print(f"- 已优化BM25参数 (k1=0.5, b=0.6)")
    print(f"- 同义词词典和查询扩展配置已保存")
    print(f"- 评估脚本会自动使用这些优化")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

