#!/usr/bin/env python3
"""
测试复杂的Blender脚本生成场景
从复杂需求中提取API需求并评估BM25和向量检索性能
"""
import json
import pickle
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


def tokenize_text(text: str) -> List[str]:
    """与build_bm25_index.py相同的分词函数"""
    tokens = []
    words = re.findall(r'\b\w+\b', text.lower())
    
    for word in words:
        tokens.append(word)
        if '_' in word:
            sub_words = word.split('_')
            tokens.extend(sub_words)
        camel_parts = re.findall(r'[a-z]+|[A-Z][a-z]*', word)
        if len(camel_parts) > 1:
            tokens.extend([p.lower() for p in camel_parts])
    
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen and len(token) > 1:
            seen.add(token)
            unique_tokens.append(token)
    
    return unique_tokens


def load_bm25_index(index_dir: str):
    """加载BM25索引"""
    index_path = Path(index_dir)
    
    with open(index_path / "bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    with open(index_path / "bm25_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(index_path / "bm25_texts.json", 'r') as f:
        texts = json.load(f)
    
    return bm25, metadata, texts


def load_original_docs(docs_file: str) -> List[Dict]:
    """加载原始文档"""
    with open(docs_file, 'r') as f:
        return json.load(f)


def load_vector_index(index_dir: str):
    """加载向量索引"""
    import faiss
    
    index_path = Path(index_dir)
    
    # 加载FAISS索引
    faiss_index = faiss.read_index(str(index_path / "faiss_index.bin"))
    
    # 加载embeddings
    embeddings = np.load(str(index_path / "embeddings.npy"))
    
    # 加载元数据
    with open(index_path / "api_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # 加载texts
    with open(index_path / "api_texts.jsonl", 'r') as f:
        texts = []
        for line in f:
            if line.strip():
                data = json.loads(line)
                texts.append(data['text_for_embedding'])
    
    return faiss_index, embeddings, metadata, texts


def load_embedding_model(model_dir: str):
    """加载embedding模型"""
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return tokenizer, model, device


def search_vector(query: str, tokenizer, model, device, faiss_index, top_k: int = 10):
    """使用向量检索"""
    import torch
    
    # 预处理查询
    query_processed = ' '.join(query.split())
    
    # 向量化查询
    inputs = tokenizer([query_processed], padding=True, truncation=True, 
                      max_length=512, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            query_embedding = outputs.pooler_output
        else:
            query_embedding = outputs.last_hidden_state[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        query_embedding = query_embedding.cpu().numpy()
    
    # FAISS搜索
    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]


def search_apis_by_keywords(docs: List[Dict], keywords: List[str]) -> List[Dict]:
    """
    通过关键词在原始文档中搜索相关API
    """
    results = []
    keywords_lower = [k.lower() for k in keywords]
    
    for doc in docs:
        doc_text = (
            doc.get('full_name', '') + ' ' + 
            doc.get('name', '') + ' ' + 
            doc.get('description', '')
        ).lower()
        
        # 检查是否包含所有关键词
        if all(keyword in doc_text for keyword in keywords_lower):
            results.append(doc)
    
    return results


def analyze_complex_requirement():
    """
    分析复杂需求，提取所需的API操作
    """
    requirement = """
    在Blender中，请你通过Python API编写一个脚本，该脚本需要首先创建一个包含多个细分环切的经纬球，
    并为其应用一个基于几何节点生成的程序化纹理材质，该材质应能根据顶点坐标动态混合两种颜色；
    接着，你需要场景中创建一个贝塞尔曲线路径，并让一个新建的棱角球体沿着这条路径执行复杂的动画，
    其中路径动画的帧范围、缓动类型以及循环模式都需要通过API精确设定；
    最后，请为整个场景设置基于物理的灯光和环境纹理，并确保将渲染引擎切换为Cycles，
    同时将所有操作整合在一个函数中，该函数需包含必要的异常处理以确保在缺少默认场景对象时能稳定运行。
    """
    
    print("复杂需求分析:")
    print("=" * 60)
    print(requirement)
    print("=" * 60)
    
    # 提取核心操作和对应的可能API
    operations = {
        "创建经纬球": {
            "keywords": ["uv", "sphere", "add"],
            "expected_api": "bpy.ops.mesh.primitive_uv_sphere_add",
            "queries": [
                "create uv sphere",
                "add uv sphere mesh",
                "primitive uv sphere add",
                "create sphere with segments"
            ]
        },
        "创建棱角球体": {
            "keywords": ["ico", "sphere", "add"],
            "expected_api": "bpy.ops.mesh.primitive_ico_sphere_add",
            "queries": [
                "create ico sphere",
                "add icosphere",
                "primitive ico sphere add",
                "create icosahedral sphere"
            ]
        },
        "创建贝塞尔曲线": {
            "keywords": ["bezier", "curve", "add"],
            "expected_api": "bpy.ops.curve.primitive_bezier_curve_add",
            "queries": [
                "create bezier curve",
                "add bezier curve path",
                "primitive bezier curve add",
                "create curve path"
            ]
        },
        "添加材质槽": {
            "keywords": ["material", "slot", "add"],
            "expected_api": "bpy.ops.object.material_slot_add",
            "queries": [
                "add material slot",
                "create material slot",
                "add material to object"
            ]
        },
        "添加几何节点": {
            "keywords": ["geometry", "nodes", "modifier"],
            "expected_api": "bpy.ops.object.modifier_add",
            "queries": [
                "add geometry nodes modifier",
                "add modifier geometry nodes",
                "create geometry nodes"
            ]
        },
        "添加灯光": {
            "keywords": ["light", "add"],
            "expected_api": "bpy.ops.object.light_add",
            "queries": [
                "add light to scene",
                "create light",
                "add lamp object"
            ]
        },
        "添加约束": {
            "keywords": ["constraint", "add", "follow", "path"],
            "expected_api": "bpy.ops.object.constraint_add",
            "queries": [
                "add constraint to object",
                "add follow path constraint",
                "create path constraint"
            ]
        },
        "设置关键帧": {
            "keywords": ["keyframe", "insert", "animation"],
            "expected_api": "bpy.ops.anim.keyframe_insert",
            "queries": [
                "insert keyframe",
                "add keyframe animation",
                "set animation keyframe"
            ]
        }
    }
    
    return operations


def test_retrieval_on_complex_query(
    operations: Dict,
    api_id_to_index: Dict,
    metadata: Dict,
    texts: List[str],
    bm25=None,
    vector_components=None,
    method_name: str = "Unknown"
):
    """
    通用的检索测试函数
    
    Args:
        operations: 操作定义
        api_id_to_index: API ID到索引的映射
        metadata: 元数据
        texts: 文本列表
        bm25: BM25索引（如果测试BM25）
        vector_components: (tokenizer, model, device, faiss_index)（如果测试向量）
        method_name: 方法名称
    """
    print(f"\n{'='*60}")
    print(f"{method_name}检索测试")
    print(f"{'='*60}")
    
    # 创建API ID到索引的映射
    api_id_to_index = {}
    for idx, mapping in enumerate(metadata['api_mappings']):
        api_id_to_index[mapping['api_id']] = idx
    
    # 首先验证期望的API是否存在
    print("\n步骤1: 验证期望API在文档中的存在性")
    print("-" * 60)
    for op_name, op_info in operations.items():
        expected_api = op_info['expected_api']
        keywords = op_info['keywords']
        
        # 在原始文档中搜索
        matches = search_apis_by_keywords(original_docs, keywords)
        
        print(f"\n操作: {op_name}")
        print(f"  期望API: {expected_api}")
        print(f"  关键词: {keywords}")
        print(f"  找到匹配: {len(matches)} 个")
        
        # 显示前3个匹配
        for i, match in enumerate(matches[:3], 1):
            print(f"    {i}. {match['full_name']}")
            if match['full_name'] == expected_api:
                print(f"       ✓ 匹配期望API！")
                op_info['verified'] = True
            else:
                print(f"       描述: {match['description'][:60]}...")
        
        if not matches:
            print(f"    ⚠ 警告: 未找到匹配的API")
            op_info['verified'] = False
    
    # 测试查询检索
    print("\n" + "=" * 60)
    print("步骤2: 测试查询检索效果")
    print("=" * 60)
    
    all_results = []
    
    for op_name, op_info in operations.items():
        expected_api = op_info['expected_api']
        queries = op_info['queries']
        
        print(f"\n操作: {op_name}")
        print(f"期望API: {expected_api}")
        print("-" * 60)
        
        if expected_api not in api_id_to_index:
            print(f"  ⚠ API不在索引中，跳过")
            continue
        
        correct_index = api_id_to_index[expected_api]
        
        query_results = []
        
        for query in queries:
            print(f"\n  查询: \"{query}\"")
            
            # 分词
            query_tokens = tokenize_text(query)
            print(f"  分词: {query_tokens}")
            
            # BM25检索
            scores = bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), 
                               key=lambda i: scores[i], 
                               reverse=True)[:10]
            
            # 查找正确答案的排名
            correct_rank = None
            for rank, idx in enumerate(top_indices, 1):
                if idx == correct_index:
                    correct_rank = rank
                    break
            
            if correct_rank:
                print(f"  ✓ 找到！排名: 第{correct_rank}位")
            else:
                print(f"  ✗ 未在Top-10中找到")
            
            # 显示Top-3
            print(f"  Top-3结果:")
            for rank, idx in enumerate(top_indices[:3], 1):
                api_id = metadata['api_mappings'][idx]['api_id']
                score = scores[idx]
                is_correct = (idx == correct_index)
                marker = "★" if is_correct else " "
                print(f"    {marker} {rank}. [{score:.2f}] {api_id}")
            
            query_results.append({
                'query': query,
                'correct_rank': correct_rank,
                'found': correct_rank is not None
            })
        
        all_results.append({
            'operation': op_name,
            'expected_api': expected_api,
            'query_results': query_results
        })
    
    return all_results


def calculate_hit_rate(all_results: List[Dict]):
    """计算Hit Rate"""
    print("\n" + "=" * 60)
    print("Hit Rate统计")
    print("=" * 60)
    
    total_queries = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    
    operations_success = 0
    total_operations = len(all_results)
    
    for result in all_results:
        op_name = result['operation']
        query_results = result['query_results']
        
        op_has_success = False
        
        for qr in query_results:
            total_queries += 1
            rank = qr['correct_rank']
            
            if rank is not None:
                op_has_success = True
                if rank == 1:
                    hit_at_1 += 1
                if rank <= 5:
                    hit_at_5 += 1
                if rank <= 10:
                    hit_at_10 += 1
        
        if op_has_success:
            operations_success += 1
    
    print(f"\n总操作数: {total_operations}")
    print(f"总查询数: {total_queries}")
    print(f"平均每操作查询数: {total_queries/total_operations:.1f}")
    
    print(f"\n查询级别Hit Rate:")
    print(f"  Hit@1:  {hit_at_1/total_queries:.2%} ({hit_at_1}/{total_queries})")
    print(f"  Hit@5:  {hit_at_5/total_queries:.2%} ({hit_at_5}/{total_queries})")
    print(f"  Hit@10: {hit_at_10/total_queries:.2%} ({hit_at_10}/{total_queries})")
    
    print(f"\n操作级别成功率:")
    print(f"  至少一个查询成功: {operations_success/total_operations:.2%} ({operations_success}/{total_operations})")
    
    # 详细分析每个操作
    print(f"\n每个操作的详细结果:")
    print("-" * 60)
    for result in all_results:
        op_name = result['operation']
        query_results = result['query_results']
        
        success_count = sum(1 for qr in query_results if qr['found'])
        total_count = len(query_results)
        
        print(f"\n{op_name}:")
        print(f"  成功查询: {success_count}/{total_count}")
        
        for qr in query_results:
            status = f"第{qr['correct_rank']}位" if qr['found'] else "未找到"
            marker = "✓" if qr['found'] else "✗"
            print(f"    {marker} \"{qr['query'][:40]}...\" → {status}")


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    
    # 路径配置
    bm25_index_dir = base_dir / "bm25_index"
    original_docs_file = base_dir / "structured_docs" / "bpy_ops_flat.json"
    
    print("=" * 60)
    print("复杂查询场景测试")
    print("=" * 60)
    
    # 分析需求
    operations = analyze_complex_requirement()
    
    # 加载数据
    print("\n加载BM25索引和原始文档...")
    bm25, metadata, texts = load_bm25_index(str(bm25_index_dir))
    original_docs = load_original_docs(str(original_docs_file))
    print(f"索引大小: {len(metadata['api_mappings'])} 个API")
    print(f"原始文档: {len(original_docs)} 个API")
    
    # 测试检索
    all_results = test_bm25_on_complex_query(
        bm25, metadata, texts, original_docs, operations
    )
    
    # 计算Hit Rate
    calculate_hit_rate(all_results)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

