import json
import pickle
import re
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 从索引配置文件加载
SYNONYM_DICT = {}
API_PATTERNS = {}


def load_bm25_config(config_file: str):
    """加载BM25配置"""
    global SYNONYM_DICT, API_PATTERNS
    
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            SYNONYM_DICT = config.get('synonyms', {})
            API_PATTERNS = config.get('api_patterns', {})
            print(f"已加载BM25优化配置: {len(SYNONYM_DICT)}个同义词, {len(API_PATTERNS)}个API模式")


def apply_synonyms(text: str) -> str:
    """应用同义词替换"""
    text_lower = text.lower()
    for original, synonym in SYNONYM_DICT.items():
        pattern = r'\b' + re.escape(original) + r'\b'
        text_lower = re.sub(pattern, synonym, text_lower)
    return text_lower


def expand_query(query: str) -> str:
    """扩展查询"""
    query_lower = query.lower()
    expanded_terms = [query_lower]
    
    # 根据关键词添加API模式
    for keyword, patterns in API_PATTERNS.items():
        if keyword in query_lower:
            expanded_terms.extend(patterns)
    
    # 如果包含add/create，添加通用模式
    if 'add' in query_lower or 'create' in query_lower:
        words = query_lower.split()
        for word in words:
            if word not in ['add', 'create', 'the', 'to', 'a', 'an', 'in', 'on', 'with']:
                expanded_terms.append(f"{word}_add")
    
    return ' '.join(expanded_terms)


def tokenize_text(text: str, apply_synonym: bool = True) -> List[str]:
    """与build_bm25_index.py相同的分词函数"""
    if apply_synonym and SYNONYM_DICT:
        text = apply_synonyms(text)
    
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


def load_vector_index(index_dir: str):
    """加载向量索引"""
    import faiss
    
    index_path = Path(index_dir)
    
    # 加载FAISS索引
    faiss_index = faiss.read_index(str(index_path / "faiss_index.bin"))
    
    # 加载embeddings
    embeddings = np.load(str(index_path / "embeddings.npy"))
    
    # 加载metadata
    with open(index_path / "api_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # 加载配置
    with open(index_path / "config.json", 'r') as f:
        config = json.load(f)
    
    return faiss_index, embeddings, metadata, config


def load_vector_model(model_dir: str):
    """加载向量化模型"""
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    model_path = Path(model_dir)
    
    # 加载tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return tokenizer, model, device


def load_reranker_model(model_dir: str):
    """加载bge-reranker-large模型"""
    model_path = Path(model_dir) / "bge-reranker-large"
    
    print(f"加载重排模型: {model_path}")
    
    # 检查模型是否存在，如果不存在则下载
    if not model_path.exists():
        print("下载bge-reranker-large模型...")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 下载模型 - 使用正确的模型类型
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
        
        # 保存到本地
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        print("模型下载完成")
    else:
        print("从本地加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        # 确保使用正确的模型类型
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"重排模型已加载到设备: {device}")
    return tokenizer, model, device



def load_queries_from_json(json_file: str) -> Dict:
    """从queries.json文件加载测试用例"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def test_bm25_retrieval(
    operations: Dict,
    bm25,
    metadata: Dict,
    texts: List[str],
    verbose: bool = False,
    use_query_expansion: bool = True
) -> Tuple[List[Dict], float]:
    """测试BM25检索（优化版）并返回平均用时"""
    api_id_to_index = {m['api_id']: i for i, m in enumerate(metadata['api_mappings'])}
    
    all_results = []
    total_time = 0.0
    query_count = 0
    
    for op_name, op_info in operations.items():
        expected_api = op_info['expected_api']
        queries = op_info['queries']
        
        if expected_api not in api_id_to_index:
            continue
        
        correct_index = api_id_to_index[expected_api]
        query_results = []
        
        for query in queries:
            # 高精度时间测量
            start_time = time.perf_counter()
            
            # 查询扩展
            if use_query_expansion:
                expanded = expand_query(query)
                query_to_use = expanded
            else:
                query_to_use = query
            
            query_tokens = tokenize_text(query_to_use)
            scores = bm25.get_scores(query_tokens)
            # 返回更多结果用于融合
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:50]
            
            end_time = time.perf_counter()
            query_time = end_time - start_time
            total_time += query_time
            query_count += 1
            
            correct_rank = None
            for rank, idx in enumerate(top_indices, 1):
                if idx == correct_index:
                    correct_rank = rank
                    break
            
            # 保存更多结果用于融合
            top_results = [
                {
                    'api_id': metadata['api_mappings'][idx]['api_id'],
                    'score': float(scores[idx])
                }
                for idx in top_indices
            ]
            
            query_results.append({
                'query': query,
                'correct_rank': correct_rank,
                'found': correct_rank is not None,
                'top_3': top_results[:3],  # 保持向后兼容
                'all_results': top_results,  # 保存所有结果用于融合
                'query_time': query_time
            })
        
        all_results.append({
            'operation': op_name,
            'expected_api': expected_api,
            'query_results': query_results
        })
    
    avg_time = total_time / query_count if query_count > 0 else 0.0
    return all_results, avg_time


def test_vector_retrieval(
    operations: Dict,
    faiss_index,
    vector_metadata: Dict,
    tokenizer,
    model,
    device: str,
    verbose: bool = False
) -> Tuple[List[Dict], float]:
    """测试向量检索并返回平均用时"""
    api_id_to_index = {m['api_id']: i for i, m in enumerate(vector_metadata['api_mappings'])}
    
    all_results = []
    total_time = 0.0
    query_count = 0
    
    for op_name, op_info in operations.items():
        expected_api = op_info['expected_api']
        queries = op_info['queries']
        
        if expected_api not in api_id_to_index:
            continue
        
        correct_index = api_id_to_index[expected_api]
        query_results = []
        
        for query in queries:
            # 高精度时间测量
            start_time = time.perf_counter()
            
            # 向量化查询
            import torch
            inputs = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    query_embedding = outputs.pooler_output
                else:
                    query_embedding = outputs.last_hidden_state[:, 0, :]
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                query_embedding = query_embedding.cpu().numpy()
            
            # 搜索更多结果用于融合
            search_k = 50
            scores, indices = faiss_index.search(query_embedding.astype('float32'), search_k)
            
            end_time = time.perf_counter()
            query_time = end_time - start_time
            total_time += query_time
            query_count += 1
            
            correct_rank = None
            for rank, idx in enumerate(indices[0], 1):
                if idx == correct_index:
                    correct_rank = rank
                    break
            
            # 保存更多结果用于融合
            vector_search_results = [
                {
                    'api_id': vector_metadata['api_mappings'][idx]['api_id'],
                    'score': float(scores[0][i])
                }
                for i, idx in enumerate(indices[0])
            ]
            
            query_results.append({
                'query': query,
                'correct_rank': correct_rank,
                'found': correct_rank is not None,
                'top_3': vector_search_results[:3],  # 保持向后兼容
                'all_results': vector_search_results,  # 保存所有结果用于融合
                'query_time': query_time
            })
        
        all_results.append({
            'operation': op_name,
            'expected_api': expected_api,
            'query_results': query_results
        })
    
    avg_time = total_time / query_count if query_count > 0 else 0.0
    return all_results, avg_time


def reciprocal_rank_fusion(bm25_results: List[Dict], vector_results: List[Dict], k: int = 60, 
                          bm25_weight: float = 0.4, vector_weight: float = 0.6) -> List[Dict]:
    """
    倒数排名融合 (Reciprocal Rank Fusion) - 优化版本，支持权重调整
    
    Args:
        bm25_results: BM25检索结果
        vector_results: 向量检索结果
        k: 融合参数，控制融合的深度
        bm25_weight: BM25检索的权重（默认0.4）
        vector_weight: 向量检索的权重（默认0.6，因为语义检索通常更准确）
    
    Returns:
        融合后的结果
    """
    fusion_results = []
    
    for bm25_result, vector_result in zip(bm25_results, vector_results):
        # 确保操作名称匹配
        assert bm25_result['operation'] == vector_result['operation']
        assert bm25_result['expected_api'] == vector_result['expected_api']
        
        fusion_query_results = []
        
        for bm25_qr, vector_qr in zip(bm25_result['query_results'], vector_result['query_results']):
            # 确保查询匹配
            assert bm25_qr['query'] == vector_qr['query']
            
            # 创建API到融合分数的映射
            api_scores = {}
            expected_api = bm25_result['expected_api']
            
            # 处理BM25结果 - 使用all_results进行融合，应用权重
            if 'all_results' in bm25_qr:
                # 使用k参数控制融合深度，但不超过实际结果数量
                fusion_depth = min(k, len(bm25_qr['all_results']))
                for rank, result in enumerate(bm25_qr['all_results'][:fusion_depth], 1):
                    api_id = result['api_id']
                    if api_id not in api_scores:
                        api_scores[api_id] = 0.0
                    # 使用加权的RRF公式：weight * 1/(k + rank)
                    api_scores[api_id] += bm25_weight * (1.0 / (k + rank))
            
            # 处理向量检索结果 - 使用all_results进行融合，应用权重
            if 'all_results' in vector_qr:
                # 使用k参数控制融合深度，但不超过实际结果数量
                fusion_depth = min(k, len(vector_qr['all_results']))
                for rank, result in enumerate(vector_qr['all_results'][:fusion_depth], 1):
                    api_id = result['api_id']
                    if api_id not in api_scores:
                        api_scores[api_id] = 0.0
                    # 使用加权的RRF公式：weight * 1/(k + rank)
                    api_scores[api_id] += vector_weight * (1.0 / (k + rank))
            
            # 按融合分数排序
            sorted_apis = sorted(api_scores.keys(), key=lambda x: api_scores[x], reverse=True)
            
            # 找到期望API的融合排名
            fusion_rank = None
            if expected_api in sorted_apis:
                fusion_rank = sorted_apis.index(expected_api) + 1
            
            # 创建all_results字段用于重排
            all_results = [{'api_id': api, 'score': api_scores[api]} for api in sorted_apis]
            
            fusion_query_results.append({
                'query': bm25_qr['query'],
                'correct_rank': fusion_rank,
                'found': fusion_rank is not None,
                'fusion_score': api_scores.get(expected_api, 0.0),
                'bm25_rank': bm25_qr['correct_rank'],
                'vector_rank': vector_qr['correct_rank'],
                'top_3': [{'api_id': api, 'score': api_scores[api]} for api in sorted_apis[:3]],
                'all_results': all_results  # 添加all_results字段
            })
        
        fusion_results.append({
            'operation': bm25_result['operation'],
            'expected_api': bm25_result['expected_api'],
            'query_results': fusion_query_results
        })
    
    return fusion_results


def rerank_candidates(query: str, candidates: List[Dict], texts: List[str], 
                     reranker_tokenizer, reranker_model, device: str, 
                     top_k: int = 3, debug_file: str = None) -> List[Dict]:
    """
    使用bge-reranker-large对候选结果进行重排
    
    Args:
        query: 查询文本
        candidates: 候选结果列表，包含api_id和score
        texts: 完整文档文本列表
        reranker_tokenizer: 重排模型分词器
        reranker_model: 重排模型
        device: 设备
        top_k: 返回的top-k结果
    
    Returns:
        重排后的结果列表
    """
    if not candidates:
        return []
    
    # 直接使用融合检索的结果，不进行预过滤
    filtered_candidates = candidates
    
    if debug_file:
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"融合候选数: {len(candidates)}\n")
    
    # 准备查询-文档对
    pairs = []
    candidate_indices = []
    
    for candidate in filtered_candidates:
        api_id = candidate['api_id']
        # 找到对应的文档索引 - 改进匹配逻辑
        doc_idx = None
        for i, text in enumerate(texts):
            # 更精确的匹配：确保API ID在文档开头或作为独立标识符出现
            if text.startswith(api_id + ':') or text.startswith(api_id + '(') or f" {api_id}:" in text:
                doc_idx = i
                break
            # 如果精确匹配失败，回退到简单匹配
            elif api_id in text:
                doc_idx = i
                break
        
        if doc_idx is not None:
            # 修正：使用元组而不是列表
            pairs.append((query, texts[doc_idx]))  # 使用元组格式
            candidate_indices.append((doc_idx, candidate))
            if debug_file:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write(f"找到文档匹配: {api_id} -> 文档长度: {len(texts[doc_idx])}\n")
        else:
            if debug_file:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write(f"警告: 未找到API {api_id} 对应的文档\n")
    
    if not pairs:
        return candidates[:top_k]
    
    # 批量处理查询-文档对
    with torch.no_grad():
        # bge-reranker-large需要特殊的输入格式：[CLS] query [SEP] document [SEP]
        # 将查询和文档组合成正确的格式
        query_doc_pairs = []
        for query, doc in pairs:
            # 使用bge-reranker的正确输入格式
            # 确保文档长度适中，避免截断重要信息
            if len(doc) > 400:  # 为查询预留空间
                doc = doc[:400] + "..."
            query_doc_pairs.append([query, doc])
        
        # 使用正确的tokenizer参数
        inputs = reranker_tokenizer(
            query_doc_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取重排分数
        outputs = reranker_model(**inputs)
        
        # 调试：保存输出信息到文件
        if debug_file:
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"重排模型输出类型: {type(outputs)}\n")
                if hasattr(outputs, 'logits'):
                    f.write(f"logits形状: {outputs.logits.shape}\n")
                f.write(f"输入对数量: {len(query_doc_pairs)}\n")
                f.write(f"前3个输入对示例:\n")
                for i, (q, d) in enumerate(query_doc_pairs[:3]):
                    f.write(f"  {i+1}. 查询: '{q[:50]}...', 文档: '{d[:50]}...'\n")
        
        # bge-reranker-large输出logits，需要正确处理
        # 对于二分类任务，logits表示相关性分数
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            # 如果是二分类，取正类的logits作为相关性分数
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1).cpu().numpy()
            else:
                # 如果是多分类，取最大值的logits
                scores = torch.max(logits, dim=-1)[0].cpu().numpy()
        else:
            # 如果没有logits，使用其他输出
            scores = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        if debug_file:
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"分数范围: {scores.min():.4f} - {scores.max():.4f}\n")
                f.write(f"前5个分数: {scores[:5]}\n")
                f.write(f"分数统计 - 均值: {scores.mean():.4f}, 标准差: {scores.std():.4f}\n")
    
    # 按重排分数排序
    scored_candidates = []
    for i, (doc_idx, candidate) in enumerate(candidate_indices):
        # 确保scores[i]是标量值
        score_value = scores[i]
        
        # 处理不同形状的数组
        if hasattr(score_value, 'shape'):
            if score_value.shape == ():
                # 标量
                score_value = float(score_value)
            elif len(score_value.shape) == 1 and score_value.shape[0] == 1:
                # 单元素数组
                score_value = float(score_value[0])
            else:
                # 多维数组，取第一个元素
                score_value = float(score_value.flat[0])
        elif hasattr(score_value, '__len__'):
            if len(score_value) == 1:
                score_value = float(score_value[0])
            else:
                score_value = float(score_value[0])
        else:
            score_value = float(score_value)
        
        scored_candidates.append({
            'api_id': candidate['api_id'],
            'rerank_score': score_value,
            'original_score': candidate.get('score', 0.0)
        })
    
    # 按重排分数降序排序
    scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # 添加重排结果的调试信息
    if debug_file:
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== 重排结果 ===\n")
            f.write(f"查询: {query}\n")
            f.write(f"候选数量: {len(scored_candidates)}\n")
            f.write(f"前{min(top_k, len(scored_candidates))}个重排结果:\n")
            for i, result in enumerate(scored_candidates[:top_k], 1):
                f.write(f"  {i}. {result['api_id']} - 重排分数: {result['rerank_score']:.4f}, 原始分数: {result['original_score']:.4f}\n")
            f.write("=" * 50 + "\n\n")
    
    return scored_candidates[:top_k]


def test_reranked_retrieval(
    operations: Dict,
    fusion_results: List[Dict],
    texts: List[str],
    reranker_tokenizer,
    reranker_model,
    device: str,
    verbose: bool = False,
    debug_file: str = None,
    fusion_top_k: int = 10
) -> Tuple[List[Dict], float]:
    """测试重排后的检索功能 - 基于融合检索结果"""
    all_results = []
    total_time = 0.0
    query_count = 0
    
    for fusion_result in fusion_results:
        expected_api = fusion_result['expected_api']
        query_results = []
        
        for fusion_qr in fusion_result['query_results']:
            # 高精度时间测量
            start_time = time.perf_counter()
            
            # 从融合结果中提取候选（前10个）
            candidates = []
            if 'all_results' in fusion_qr:
                for result in fusion_qr['all_results'][:fusion_top_k]:
                    candidates.append({
                        'api_id': result['api_id'],
                        'score': result['score'],
                        'source': 'fusion'
                    })
            
            # 重排候选结果
            reranked_results = rerank_candidates(
                fusion_qr['query'], 
                candidates, 
                texts,
                reranker_tokenizer,
                reranker_model,
                device,
                top_k=3,
                debug_file=debug_file
            )
            
            end_time = time.perf_counter()
            query_time = end_time - start_time
            total_time += query_time
            query_count += 1
            
            # 找到期望API的重排排名
            rerank_rank = None
            for rank, result in enumerate(reranked_results, 1):
                if result['api_id'] == expected_api:
                    rerank_rank = rank
                    break
            
            query_results.append({
                'query': fusion_qr['query'],
                'correct_rank': rerank_rank,
                'found': rerank_rank is not None,
                'rerank_score': reranked_results[0]['rerank_score'] if reranked_results else 0.0,
                'top_3': reranked_results[:3],
                'query_time': query_time
            })
        
        all_results.append({
            'operation': fusion_result['operation'],
            'expected_api': fusion_result['expected_api'],
            'query_results': query_results
        })
    
    avg_time = total_time / query_count if query_count > 0 else 0.0
    return all_results, avg_time


def calculate_case_metrics(all_results: List[Dict], case_operations: List[str], case_name: str = None) -> Dict:
    """计算特定case的指标"""
    # 过滤出属于该case的操作结果
    if case_name:
        # 使用唯一键匹配：case_name.op_name
        unique_keys = [f"{case_name}.{op}" for op in case_operations]
        case_results = [result for result in all_results if result['operation'] in unique_keys]
    else:
        # 向后兼容：如果没有提供case_name，使用原始逻辑
        case_results = [result for result in all_results if result['operation'] in case_operations]
    
    if not case_results:
        return {
            'total_operations': 0,
            'total_queries': 0,
            'hit_at_1': 0,
            'hit_at_3': 0,  # 修复：添加hit_at_3
            'hit_at_5': 0,
            'operations_success': 0,
            'hit_rate_at_1': 0.0,
            'hit_rate_at_3': 0.0,  # 修复：添加hit_rate_at_3
            'hit_rate_at_5': 0.0,
            'operation_success_rate': 0.0,
            'recall': 0.0,
            'recall_at_3': 0.0,  # 修复：添加recall_at_3
            'recall_at_5': 0.0,
            'precision': 0.0,
            'f1_score': 0.0,
            'mrr': 0.0,
            'success_rate': 0.0
        }
    
    return calculate_metrics(case_results)


def calculate_case_avg_time(all_results: List[Dict], case_operations: List[str], case_name: str = None) -> float:
    """计算特定case的平均用时"""
    # 过滤出属于该case的操作结果
    if case_name:
        # 使用唯一键匹配：case_name.op_name
        unique_keys = [f"{case_name}.{op}" for op in case_operations]
        case_results = [result for result in all_results if result['operation'] in unique_keys]
    else:
        # 向后兼容：如果没有提供case_name，使用原始逻辑
        case_results = [result for result in all_results if result['operation'] in case_operations]
    
    if not case_results:
        return 0.0
    
    total_time = 0.0
    query_count = 0
    
    for result in case_results:
        for qr in result['query_results']:
            if 'query_time' in qr:
                total_time += qr['query_time']
                query_count += 1
    
    return total_time / query_count if query_count > 0 else 0.0


def calculate_metrics(all_results: List[Dict]) -> Dict:
    """计算评估指标（包含召回率、MRR和成功率）"""
    total_queries = 0
    hit_at_1 = 0
    hit_at_3 = 0  # 修复：添加hit_at_3计数器
    hit_at_5 = 0
    operations_success = 0
    
    # 计算召回率相关指标
    total_relevant = 0  # 总的相关文档数
    total_retrieved = 0  # 总的检索文档数
    relevant_retrieved = 0  # 检索到的相关文档数
    
    # 计算MRR
    reciprocal_ranks = []
    
    for result in all_results:
        op_has_success = False
        
        for qr in result['query_results']:
            total_queries += 1
            rank = qr['correct_rank']
            
            # 每个查询有1个相关文档
            total_relevant += 1
            
            # 每个查询检索10个文档
            total_retrieved += 10
            
            if rank is not None:
                op_has_success = True
                relevant_retrieved += 1
                # 计算倒数排名
                reciprocal_ranks.append(1.0 / rank)
                
                if rank == 1:
                    hit_at_1 += 1
                if rank <= 3:  # 修复：添加hit_at_3的统计
                    hit_at_3 += 1
                if rank <= 5:
                    hit_at_5 += 1
            else:
                # 如果没有找到，倒数排名为0
                reciprocal_ranks.append(0.0)
        
        if op_has_success:
            operations_success += 1
    
    # 计算召回率
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
    recall_at_5 = hit_at_5 / total_queries if total_queries > 0 else 0
    recall_at_3 = hit_at_3 / total_queries if total_queries > 0 else 0  # 修复：使用正确的hit_at_3
    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算MRR
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # 计算成功率（找到正确答案的查询比例）
    success_rate = relevant_retrieved / total_queries if total_queries > 0 else 0
    
    return {
        'total_operations': len(all_results),
        'total_queries': total_queries,
        'hit_at_1': hit_at_1,
        'hit_at_3': hit_at_3,  # 修复：添加hit_at_3到返回值
        'hit_at_5': hit_at_5,
        'operations_success': operations_success,
        'hit_rate_at_1': hit_at_1 / total_queries if total_queries > 0 else 0,
        'hit_rate_at_3': hit_at_3 / total_queries if total_queries > 0 else 0,  # 修复：添加hit_rate_at_3
        'hit_rate_at_5': hit_at_5 / total_queries if total_queries > 0 else 0,
        'operation_success_rate': operations_success / len(all_results) if all_results else 0,
        'recall': recall,
        'recall_at_5': recall_at_5,
        'recall_at_3': recall_at_3,
        'precision': precision,
        'f1_score': f1_score,
        'mrr': mrr,
        'success_rate': success_rate
    }


def print_metrics(metrics: Dict, method_name: str, avg_time: float = None):
    """打印指标（包含召回率和时间）"""
    print(f"\n{method_name} 评估指标:")
    print("-" * 60)
    print(f"总操作数: {metrics['total_operations']}")
    print(f"总查询数: {metrics['total_queries']}")
    print(f"平均每操作查询数: {metrics['total_queries']/metrics['total_operations']:.1f}")
    if avg_time is not None:
        print(f"单次操作平均用时: {avg_time*1000:.6f} 毫秒")
    print(f"\n命中率:")
    print(f"  Hit@1:  {metrics['hit_rate_at_1']:.2%} ({metrics['hit_at_1']}/{metrics['total_queries']})")
    if 'hit_at_3' in metrics:  # 修复：显示Hit@3
        print(f"  Hit@3:  {metrics['hit_rate_at_3']:.2%} ({metrics['hit_at_3']}/{metrics['total_queries']})")
    print(f"  Hit@5:  {metrics['hit_rate_at_5']:.2%} ({metrics['hit_at_5']}/{metrics['total_queries']})")
    print(f"  成功率: {metrics['success_rate']:.2%}")
    print(f"\n召回率指标:")
    if 'recall_at_3' in metrics:
        print(f"  Recall@3:  {metrics['recall_at_3']:.2%}")
    print(f"  Recall@5:  {metrics['recall_at_5']:.2%}")
    print(f"  Recall@10: {metrics['recall']:.2%}")
    print(f"  精确率 (Precision): {metrics['precision']:.2%}")
    print(f"  F1分数: {metrics['f1_score']:.2%}")
    print(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
    print(f"\n操作级别成功率: {metrics['operation_success_rate']:.2%} ({metrics['operations_success']}/{metrics['total_operations']})")


def print_case_metrics(case_name: str, bm25_metrics: Dict, vector_metrics: Dict, fusion_metrics: Dict, 
                      bm25_avg_time: float, vector_avg_time: float):
    """打印特定case的指标对比"""
    print(f"\n{case_name} 指标对比:")
    print("-" * 100)
    print(f"{'方法':<12} {'Hit@1':<8} {'Hit@5':<8} {'成功率':<8} {'Recall@5':<8} {'Recall@10':<8} {'MRR':<8} {'用时(ms)':<10}")
    print("-" * 110)
    print(f"{'BM25':<12} {bm25_metrics['hit_rate_at_1']:.2%}   {bm25_metrics['hit_rate_at_5']:.2%}   {bm25_metrics['success_rate']:.2%}   {bm25_metrics['recall_at_5']:.2%}   {bm25_metrics['recall']:.2%}   {bm25_metrics['mrr']:.4f}   {bm25_avg_time*1000:.6f}")
    print(f"{'向量检索':<12} {vector_metrics['hit_rate_at_1']:.2%}   {vector_metrics['hit_rate_at_5']:.2%}   {vector_metrics['success_rate']:.2%}   {vector_metrics['recall_at_5']:.2%}   {vector_metrics['recall']:.2%}   {vector_metrics['mrr']:.4f}   {vector_avg_time*1000:.6f}")
    print(f"{'融合检索':<12} {fusion_metrics['hit_rate_at_1']:.2%}   {fusion_metrics['hit_rate_at_5']:.2%}   {fusion_metrics['success_rate']:.2%}   {fusion_metrics['recall_at_5']:.2%}   {fusion_metrics['recall']:.2%}   {fusion_metrics['mrr']:.4f}   {'N/A':<10}")


def print_operation_summary(all_results: List[Dict], method_name: str):
    """打印每个操作的摘要"""
    print(f"\n{method_name} 每个操作详细结果:")
    print("-" * 60)
    
    for result in all_results:
        op_name = result['operation']
        query_results = result['query_results']
        
        success_count = sum(1 for qr in query_results if qr['found'])
        total_count = len(query_results)
        
        print(f"\n{op_name}: {success_count}/{total_count} 成功")
        
        for qr in query_results:
            status = f"第{qr['correct_rank']}位" if qr['found'] else "未找到"
            marker = "✓" if qr['found'] else "✗"
            query_short = qr['query'][:35] + "..." if len(qr['query']) > 35 else qr['query']
            print(f"  {marker} \"{query_short}\" → {status}")


def save_detailed_results_v2(output_file: str, queries_data: Dict,
                            bm25_optimized_results: List[Dict], bm25_optimized_metrics: Dict, bm25_avg_time: float,
                            vector_results: List[Dict], vector_metrics: Dict, vector_avg_time: float,
                            fusion_results: List[Dict], fusion_metrics: Dict):
    """保存详细结果到文件（按case组织结果）"""
    
    # 按case组织结果
    case_results = {}
    
    for case_name, case_data in queries_data['cases'].items():
            
        case_operations = list(case_data['operations'].keys())
        
        # 计算该case的指标（传入case_name以正确匹配操作）
        bm25_case_metrics = calculate_case_metrics(bm25_optimized_results, case_operations, case_name)
        vector_case_metrics = calculate_case_metrics(vector_results, case_operations, case_name)
        fusion_case_metrics = calculate_case_metrics(fusion_results, case_operations, case_name)
        
        # 计算该case的独立用时（传入case_name以正确匹配操作）
        bm25_case_time = calculate_case_avg_time(bm25_optimized_results, case_operations, case_name)
        vector_case_time = calculate_case_avg_time(vector_results, case_operations, case_name)
        
        # 组织该case的结果
        case_results[case_name] = {
            'description': case_data['description'],
            'operations': {},
            'metrics_comparison': {
                'bm25': {
                    'metrics': bm25_case_metrics,
                    'avg_time_seconds': bm25_case_time,
                    'avg_time_milliseconds': bm25_case_time * 1000
                },
                'vector': {
                    'metrics': vector_case_metrics,
                    'avg_time_seconds': vector_case_time,
                    'avg_time_milliseconds': vector_case_time * 1000
                },
                'fusion': {
                    'metrics': fusion_case_metrics
                }
            }
        }
        
        # 为每个操作添加简化结果（不保存详细的query_results）
        for op_name in case_operations:
            # 查找该操作在结果中的索引（使用唯一键：case_name.op_name）
            unique_op_key = f"{case_name}.{op_name}"
            bm25_op_result = next((r for r in bm25_optimized_results if r['operation'] == unique_op_key), None)
            vector_op_result = next((r for r in vector_results if r['operation'] == unique_op_key), None)
            fusion_op_result = next((r for r in fusion_results if r['operation'] == unique_op_key), None)
            
            # 只保存统计信息，不保存详细的query_results
            case_results[case_name]['operations'][op_name] = {
                'expected_api': case_data['operations'][op_name]['expected_api'],
                'queries_count': len(case_data['operations'][op_name]['queries']),
                'bm25_success_count': sum(1 for qr in bm25_op_result['query_results'] if qr['found']) if bm25_op_result else 0,
                'vector_success_count': sum(1 for qr in vector_op_result['query_results'] if qr['found']) if vector_op_result else 0,
                'fusion_success_count': sum(1 for qr in fusion_op_result['query_results'] if qr['found']) if fusion_op_result else 0
            }
    
    # 构建最终数据结构
    data = {
        'timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'bm25_optimized': {
                'description': 'BM25优化版本：启用同义词和查询扩展',
                'metrics': bm25_optimized_metrics,
                'avg_time_seconds': bm25_avg_time,
                'avg_time_milliseconds': bm25_avg_time * 1000
            },
            'vector_retrieval': {
                'description': '向量检索：基于sentence-transformers和FAISS',
                'metrics': vector_metrics,
                'avg_time_seconds': vector_avg_time,
                'avg_time_milliseconds': vector_avg_time * 1000
            },
            'fusion_retrieval': {
                'description': '倒数排名融合：BM25 + 向量检索',
                'metrics': fusion_metrics
            }
        },
        'case_results': case_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    
    # 路径配置
    bm25_index_dir = base_dir / "bm25_index"
    vector_index_dir = base_dir / "vector_index"
    model_dir = base_dir / "all-MiniLM-L6-v2"
    output_file = base_dir / "complex_query_test_results.json"
    
    print("=" * 60)
    print("复杂查询场景测试 - BM25优化 + 向量检索")
    print("=" * 60)
    
    # 加载测试用例
    queries_data = load_queries_from_json(str(base_dir / "queries.json"))
    print(f"测试用例数: {len(queries_data['cases'])}")
    
    # 修复：为每个操作添加case前缀，避免不同case的操作互相覆盖
    all_operations = {}
    for case_name, case_data in queries_data['cases'].items():
        for op_name, op_data in case_data['operations'].items():
            # 使用 case_name.op_name 作为唯一键
            unique_key = f"{case_name}.{op_name}"
            all_operations[unique_key] = op_data
    
    print(f"总操作数: {len(all_operations)}")
    
    # 加载BM25配置
    config_file = bm25_index_dir / "bm25_config.json"
    load_bm25_config(str(config_file))
    
    # 加载BM25索引
    print("\n加载BM25索引...")
    bm25, bm25_metadata, bm25_texts = load_bm25_index(str(bm25_index_dir))
    print(f"BM25索引: {len(bm25_metadata['api_mappings'])} 个API")
    
    # 加载向量索引
    print("\n加载向量索引...")
    faiss_index, embeddings, vector_metadata, vector_config = load_vector_index(str(vector_index_dir))
    print(f"向量索引: {len(vector_metadata['api_mappings'])} 个API")
    print(f"向量维度: {vector_config['embedding_dim']}")
    
    # 加载向量化模型
    print("\n加载向量化模型...")
    tokenizer, model, device = load_vector_model(str(model_dir))
    print(f"模型设备: {device}")
    
    # 加载重排模型
    print("\n加载重排模型...")
    reranker_tokenizer, reranker_model, reranker_device = load_reranker_model(str(base_dir))
    print(f"重排模型设备: {reranker_device}")
    
    # 测试BM25 - 优化版本
    print("\n" + "=" * 60)
    print("测试BM25检索 - 优化版本（启用同义词与查询扩展）")
    print("=" * 60)
    bm25_optimized_results, bm25_avg_time = test_bm25_retrieval(
        all_operations, bm25, bm25_metadata, bm25_texts,
        use_query_expansion=True
    )
    bm25_optimized_metrics = calculate_metrics(bm25_optimized_results)
    print_metrics(bm25_optimized_metrics, "BM25优化", bm25_avg_time)
    
    # 测试向量检索
    print("\n" + "=" * 60)
    print("测试向量检索 - 基于sentence-transformers和FAISS")
    print("=" * 60)
    vector_results, vector_avg_time = test_vector_retrieval(
        all_operations, faiss_index, vector_metadata, tokenizer, model, device
    )
    vector_metrics = calculate_metrics(vector_results)
    print_metrics(vector_metrics, "向量检索", vector_avg_time)
    
    # 倒数排名融合
    print("\n" + "=" * 60)
    print("倒数排名融合 - BM25 + 向量检索")
    print("=" * 60)
    fusion_results = reciprocal_rank_fusion(bm25_optimized_results, vector_results)
    fusion_metrics = calculate_metrics(fusion_results)
    print_metrics(fusion_metrics, "融合检索")
    
    # 重排检索
    print("\n" + "=" * 60)
    print("重排检索 - bge-reranker-large")
    print("=" * 60)
    
    # 创建调试文件
    debug_file = base_dir / "rerank_debug.log"
    if debug_file.exists():
        debug_file.unlink()  # 删除旧的调试文件
    
    reranked_results, reranked_avg_time = test_reranked_retrieval(
        all_operations, fusion_results, bm25_texts,
        reranker_tokenizer, reranker_model, reranker_device,
        fusion_top_k=30,  # 优化：增加到25个候选，提高召回率
        debug_file=str(debug_file)
    )
    reranked_metrics = calculate_metrics(reranked_results)
    print_metrics(reranked_metrics, "重排检索", reranked_avg_time)
    
    print(f"\n调试信息已保存到: {debug_file}")
    
    # 按case显示结果
    print("\n" + "=" * 60)
    print("各测试用例详细结果")
    print("=" * 60)
    
    for case_name, case_data in queries_data['cases'].items():
        case_operations = list(case_data['operations'].keys())
        
        # 计算该case的指标（传入case_name以正确匹配操作）
        bm25_case_metrics = calculate_case_metrics(bm25_optimized_results, case_operations, case_name)
        vector_case_metrics = calculate_case_metrics(vector_results, case_operations, case_name)
        fusion_case_metrics = calculate_case_metrics(fusion_results, case_operations, case_name)
        
        # 计算该case的独立用时（传入case_name以正确匹配操作）
        bm25_case_time = calculate_case_avg_time(bm25_optimized_results, case_operations, case_name)
        vector_case_time = calculate_case_avg_time(vector_results, case_operations, case_name)
        
        # 打印该case的对比结果
        print_case_metrics(case_name, bm25_case_metrics, vector_case_metrics, fusion_case_metrics, 
                          bm25_case_time, vector_case_time)
    
    # 保存详细结果（按case组织结果）
    save_detailed_results_v2(str(output_file), queries_data,
                            bm25_optimized_results, bm25_optimized_metrics, bm25_avg_time,
                            vector_results, vector_metrics, vector_avg_time,
                            fusion_results, fusion_metrics)
    
    # 对比总结
    print("\n" + "=" * 60)
    print("检索方式对比总结")
    print("=" * 60)
    print(f"BM25优化 - Hit@1: {bm25_optimized_metrics['hit_rate_at_1']:.2%}, Recall@3: {bm25_optimized_metrics['recall_at_3']:.2%}, Recall@5: {bm25_optimized_metrics['recall_at_5']:.2%}, Recall@10: {bm25_optimized_metrics['recall']:.2%}, MRR: {bm25_optimized_metrics['mrr']:.4f}, 用时: {bm25_avg_time*1000:.6f}ms")
    print(f"向量检索 - Hit@1: {vector_metrics['hit_rate_at_1']:.2%}, Recall@3: {vector_metrics['recall_at_3']:.2%}, Recall@5: {vector_metrics['recall_at_5']:.2%}, Recall@10: {vector_metrics['recall']:.2%}, MRR: {vector_metrics['mrr']:.4f}, 用时: {vector_avg_time*1000:.6f}ms")
    print(f"融合检索 - Hit@1: {fusion_metrics['hit_rate_at_1']:.2%}, Recall@3: {fusion_metrics['recall_at_3']:.2%}, Recall@5: {fusion_metrics['recall_at_5']:.2%}, Recall@10: {fusion_metrics['recall']:.2%}, MRR: {fusion_metrics['mrr']:.4f}")
    print(f"重排检索 - Hit@1: {reranked_metrics['hit_rate_at_1']:.2%}, Recall@3: {reranked_metrics['recall_at_3']:.2%}, MRR: {reranked_metrics['mrr']:.4f}, 用时: {reranked_avg_time*1000:.6f}ms")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
