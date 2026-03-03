"""
BM25 + 向量检索 + 融合检索 + 重排工具
可被上游工具调用，支持 L1/L2/L3 控制参数
"""
import json
import pickle
import re
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


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
    
    # 检测是否为属性设置查询（包含 set/adjust/configure 等动词）
    is_property_query = any(verb in query_lower for verb in ['set', 'adjust', 'configure', 'change', 'modify'])
    
    # 如果是属性查询，添加属性相关的术语
    if is_property_query:
        expanded_terms.extend(['property', 'attribute', 'value', 'parameter'])
        
        # 针对常见属性类型添加扩展
        if 'energy' in query_lower or 'power' in query_lower:
            expanded_terms.extend(['energy', 'power', 'intensity', 'strength'])
        
        if 'color' in query_lower or 'colour' in query_lower:
            expanded_terms.extend(['color', 'colour', 'tint'])
        
        if 'influence' in query_lower or 'strength' in query_lower or 'weight' in query_lower:
            expanded_terms.extend(['influence', 'strength', 'weight', 'amount', 'factor'])
        
        if 'target' in query_lower:
            expanded_terms.extend(['target', 'destination', 'object', 'reference'])
    
    # 如果包含add/create（但不是属性查询），添加通用模式
    elif 'add' in query_lower or 'create' in query_lower:
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
    """加载bge-reranker-large重排模型"""
    model_path = Path(model_dir) / "bge-reranker-large"
    
    print(f"加载重排模型: {model_path}")
    
    # 检查模型是否存在，如果不存在则下载
    if not model_path.exists():
        print("下载bge-reranker-large模型...")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
        
        # 保存到本地
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        print("模型下载完成")
    else:
        print("从本地加载重排模型...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"重排模型已加载到设备: {device}")
    return tokenizer, model, device


class HybridRetriever:
    """混合检索器：BM25 + 向量检索 + 融合 + 重排"""
    
    def __init__(self, base_dir: str, enable_reranker: bool = False):
        """
        初始化混合检索器
        
        Args:
            base_dir: 基础目录，包含索引和模型
            enable_reranker: 是否启用重排模型（默认False以保持向后兼容）
        """
        self.base_dir = Path(base_dir)
        self.enable_reranker = enable_reranker
        
        # 路径配置
        self.bm25_index_dir = self.base_dir / "bm25_index"
        self.vector_index_dir = self.base_dir / "vector_index"
        self.model_dir = self.base_dir / "all-MiniLM-L6-v2"
        
        # 加载配置
        config_file = self.bm25_index_dir / "bm25_config.json"
        load_bm25_config(str(config_file))
        
        # 加载BM25索引
        print("加载BM25索引...")
        self.bm25, self.bm25_metadata, self.bm25_texts = load_bm25_index(str(self.bm25_index_dir))
        print(f"BM25索引: {len(self.bm25_metadata['api_mappings'])} 个API")
        
        # 加载向量索引
        print("加载向量索引...")
        self.faiss_index, self.embeddings, self.vector_metadata, self.vector_config = load_vector_index(
            str(self.vector_index_dir)
        )
        print(f"向量索引: {len(self.vector_metadata['api_mappings'])} 个API")
        print(f"向量维度: {self.vector_config['embedding_dim']}")
        
        # 加载向量化模型
        print("加载向量化模型...")
        self.tokenizer, self.model, self.device = load_vector_model(str(self.model_dir))
        print(f"模型设备: {self.device}")
        
        # 加载重排模型（如果启用）
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.reranker_device = None
        if self.enable_reranker:
            print("加载重排模型...")
            self.reranker_tokenizer, self.reranker_model, self.reranker_device = load_reranker_model(str(self.base_dir))
            print(f"重排模型设备: {self.reranker_device}")
        
        print("混合检索器初始化完成\n")
    
    def bm25_search(self, query: str, top_k: int = 50, use_query_expansion: bool = True) -> List[Dict]:
        """
        BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            use_query_expansion: 是否使用查询扩展
            
        Returns:
            检索结果列表，每个元素包含 api_id 和 score
        """
        # 查询扩展
        if use_query_expansion:
            expanded = expand_query(query)
            query_to_use = expanded
        else:
            query_to_use = query
        
        # 分词和检索
        query_tokens = tokenize_text(query_to_use)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # 构建结果
        results = []
        for idx in top_indices:
            results.append({
                'api_id': self.bm25_metadata['api_mappings'][idx]['api_id'],
                'score': float(scores[idx])
            })
        
        return results
    
    def vector_search(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表，每个元素包含 api_id 和 score
        """
        # 向量化查询
        inputs = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                query_embedding = outputs.pooler_output
            else:
                query_embedding = outputs.last_hidden_state[:, 0, :]
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            query_embedding = query_embedding.cpu().numpy()
        
        # FAISS搜索
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'api_id': self.vector_metadata['api_mappings'][idx]['api_id'],
                'score': float(scores[0][i])
            })
        
        return results
    
    def reciprocal_rank_fusion(self, bm25_results: List[Dict], vector_results: List[Dict], 
                               k: int = 60) -> List[Dict]:
        """
        倒数排名融合 (Reciprocal Rank Fusion)
        
        Args:
            bm25_results: BM25检索结果
            vector_results: 向量检索结果
            k: 融合参数，控制融合的深度
            
        Returns:
            融合后的结果列表，按分数降序排列
        """
        # 创建API到融合分数的映射
        api_scores = {}
        
        # 处理BM25结果
        fusion_depth = min(k, len(bm25_results))
        for rank, result in enumerate(bm25_results[:fusion_depth], 1):
            api_id = result['api_id']
            if api_id not in api_scores:
                api_scores[api_id] = 0.0
            # 使用标准的RRF公式：1/(k + rank)
            api_scores[api_id] += 1.0 / (k + rank)
        
        # 处理向量检索结果
        fusion_depth = min(k, len(vector_results))
        for rank, result in enumerate(vector_results[:fusion_depth], 1):
            api_id = result['api_id']
            if api_id not in api_scores:
                api_scores[api_id] = 0.0
            # 使用标准的RRF公式：1/(k + rank)
            api_scores[api_id] += 1.0 / (k + rank)
        
        # 按融合分数排序
        sorted_apis = sorted(api_scores.keys(), key=lambda x: api_scores[x], reverse=True)
        
        # 构建结果列表
        fusion_results = [
            {
                'api_id': api_id,
                'score': api_scores[api_id]
            }
            for api_id in sorted_apis
        ]
        
        return fusion_results
    
    def rerank_candidates(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        使用bge-reranker-large对候选结果进行重排
        
        Args:
            query: 查询文本
            candidates: 候选结果列表，包含api_id和score
            top_k: 返回的top-k结果
            
        Returns:
            重排后的结果列表
        """
        if not candidates:
            return []
        
        if not self.enable_reranker:
            # 如果未启用重排，直接返回前top_k个候选
            return candidates[:top_k]
        
        # 准备查询-文档对
        pairs = []
        candidate_indices = []
        
        for candidate in candidates:
            api_id = candidate['api_id']
            # 找到对应的文档文本
            doc_text = None
            for i, text in enumerate(self.bm25_texts):
                # 精确匹配API ID
                if text.startswith(api_id + ':') or text.startswith(api_id + '(') or f" {api_id}:" in text:
                    doc_text = text
                    break
                # 回退到简单匹配
                elif api_id in text:
                    doc_text = text
                    break
            
            if doc_text is not None:
                # 限制文档长度以避免截断
                if len(doc_text) > 400:
                    doc_text = doc_text[:400] + "..."
                pairs.append([query, doc_text])
                candidate_indices.append(candidate)
        
        if not pairs:
            return candidates[:top_k]
        
        # 批量处理查询-文档对
        with torch.no_grad():
            # 使用bge-reranker的输入格式
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移到设备
            inputs = {k: v.to(self.reranker_device) for k, v in inputs.items()}
            
            # 获取重排分数
            outputs = self.reranker_model(**inputs)
            
            # 处理输出logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    scores = logits.squeeze(-1).cpu().numpy()
                else:
                    scores = torch.max(logits, dim=-1)[0].cpu().numpy()
            else:
                scores = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # 构建评分后的候选列表
        scored_candidates = []
        for i, candidate in enumerate(candidate_indices):
            score_value = float(scores[i]) if hasattr(scores[i], '__float__') else float(scores[i].item())
            scored_candidates.append({
                'api_id': candidate['api_id'],
                'rerank_score': score_value,
                'original_score': candidate.get('score', 0.0)
            })
        
        # 按重排分数降序排序
        scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return scored_candidates[:top_k]
    
    def exact_match_search(self, query: str) -> Optional[str]:
        """
        精确匹配搜索：直接在API列表中查找完全匹配的API ID
        
        Args:
            query: API查询字符串（如 bpy.ops.mesh.subdivide）
            
        Returns:
            匹配的API ID，如果未找到返回None
        """
        # 获取所有API ID列表
        all_api_ids = [mapping['api_id'] for mapping in self.bm25_metadata['api_mappings']]
        
        # 精确匹配
        if query in all_api_ids:
            return query
        
        # 尝试不区分大小写匹配
        query_lower = query.lower()
        for api_id in all_api_ids:
            if api_id.lower() == query_lower:
                return api_id
        
        return None
    
    def search(self, queries: List[str], level: str = "L1", 
               use_query_expansion: bool = True, fusion_k: int = 60) -> List[List[str]]:
        """
        混合检索：对多个查询进行检索（支持重排）
        
        Args:
            queries: 查询列表
            level: 控制参数，"L1" 返回top1，"L2" 返回top3，"L3" 返回top5
            use_query_expansion: 是否使用查询扩展
            fusion_k: 融合参数k
            
        Returns:
            结果列表，每个查询对应一个API列表
            - L1: 每个查询返回1个API（使用精确匹配）
            - L2: 每个查询返回3个API（如果启用重排，先融合top10再重排到top3）
            - L3: 每个查询返回5个API
        """
        # 确定返回的API数量
        top_k_map = {
            "L1": 1,
            "L2": 3,
            "L3": 5
        }
        
        if level not in top_k_map:
            raise ValueError(f"Invalid level: {level}. Must be one of: L1, L2, L3")
        
        top_k = top_k_map[level]
        
        results = []
        
        for query in queries:
            # L1模式：使用精确匹配
            if level == "L1":
                matched_api = self.exact_match_search(query)
                if matched_api:
                    top_apis = [matched_api]
                else:
                    # 如果精确匹配失败，返回空列表
                    top_apis = []
            else:
                # L2/L3模式：使用混合检索
                # BM25检索
                bm25_results = self.bm25_search(query, top_k=50, use_query_expansion=use_query_expansion)
                
                # 向量检索
                vector_results = self.vector_search(query, top_k=50)
                
                # 融合
                fusion_results = self.reciprocal_rank_fusion(bm25_results, vector_results, k=fusion_k)
                
                # 如果启用重排且level为L2，使用融合top10进行重排
                if self.enable_reranker and level == "L2":
                    # 融合提供top10候选
                    fusion_candidates = fusion_results[:10]
                    # 重排返回top3
                    reranked_results = self.rerank_candidates(query, fusion_candidates, top_k=3)
                    top_apis = [result['api_id'] for result in reranked_results]
                else:
                    # 不使用重排，直接返回融合结果的top-k
                    top_apis = [result['api_id'] for result in fusion_results[:top_k]]
            
            results.append(top_apis)
        
        return results
    
    def search_with_scores(self, queries: List[str], level: str = "L1",
                          use_query_expansion: bool = True, fusion_k: int = 60) -> List[List[Dict]]:
        """
        混合检索：对多个查询进行检索，返回带分数的结果（支持重排）
        
        Args:
            queries: 查询列表
            level: 控制参数，"L1" 返回top1，"L2" 返回top3，"L3" 返回top5
            use_query_expansion: 是否使用查询扩展
            fusion_k: 融合参数k
            
        Returns:
            结果列表，每个查询对应一个结果列表，每个结果包含 api_id 和 score (或 rerank_score)
        """
        # 确定返回的API数量
        top_k_map = {
            "L1": 1,
            "L2": 3,
            "L3": 5
        }
        
        if level not in top_k_map:
            raise ValueError(f"Invalid level: {level}. Must be one of: L1, L2, L3")
        
        top_k = top_k_map[level]
        
        results = []
        
        for query in queries:
            # L1模式：使用精确匹配
            if level == "L1":
                matched_api = self.exact_match_search(query)
                if matched_api:
                    top_results = [{'api_id': matched_api, 'score': 1.0}]
                else:
                    # 如果精确匹配失败，返回空列表
                    top_results = []
            else:
                # L2/L3模式：使用混合检索
                # BM25检索
                bm25_results = self.bm25_search(query, top_k=50, use_query_expansion=use_query_expansion)
                
                # 向量检索
                vector_results = self.vector_search(query, top_k=50)
                
                # 融合
                fusion_results = self.reciprocal_rank_fusion(bm25_results, vector_results, k=fusion_k)
                
                # 如果启用重排且level为L2，使用融合top10进行重排
                if self.enable_reranker and level == "L2":
                    # 融合提供top10候选
                    fusion_candidates = fusion_results[:10]
                    # 重排返回top3，带有rerank_score
                    top_results = self.rerank_candidates(query, fusion_candidates, top_k=3)
                else:
                    # 不使用重排，直接返回融合结果的top-k
                    top_results = fusion_results[:top_k]
            
            results.append(top_results)
        
        return results


def create_retriever(base_dir: str, enable_reranker: bool = False) -> HybridRetriever:
    """
    创建混合检索器实例
    
    Args:
        base_dir: 基础目录路径
        enable_reranker: 是否启用重排模型（默认False）
        
    Returns:
        HybridRetriever实例
    """
    return HybridRetriever(base_dir, enable_reranker=enable_reranker)


# 使用示例
if __name__ == '__main__':
    # 示例：创建检索器并执行查询
    base_dir = Path(__file__).parent
    
    print("=" * 60)
    print("混合检索器示例")
    print("=" * 60)
    
    # 创建检索器
    retriever = create_retriever(str(base_dir))
    
    # 测试查询
    test_queries = [
        "bpy.ops.object.material_slot_add",
        "bpy.ops.object.light_add",
        "bpy.ops.object.constraint_add",
    ]
    
    print(f"\n测试查询: {test_queries}\n")
    
    # L1 模式：返回top1
    print("=" * 60)
    print("L1 模式 - 返回 Top 1")
    print("=" * 60)
    results_l1 = retriever.search(test_queries, level="L1")
    for query, apis in zip(test_queries, results_l1):
        print(f"Query: {query}")
        print(f"  Top 1: {apis}")
    
    # L2 模式：返回top3
    print("\n" + "=" * 60)
    print("L2 模式 - 返回 Top 3")
    print("=" * 60)
    results_l2 = retriever.search(test_queries, level="L2")
    for query, apis in zip(test_queries, results_l2):
        print(f"Query: {query}")
        print(f"  Top 3: {apis}")
    
    # L3 模式：返回top5
    print("\n" + "=" * 60)
    print("L3 模式 - 返回 Top 5")
    print("=" * 60)
    results_l3 = retriever.search(test_queries, level="L3")
    for query, apis in zip(test_queries, results_l3):
        print(f"Query: {query}")
        print(f"  Top 5: {apis}")
    
    # 带分数的结果
    print("\n" + "=" * 60)
    print("L2 模式 - 返回 Top 3（带分数）")
    print("=" * 60)
    results_with_scores = retriever.search_with_scores(test_queries, level="L2")
    for query, results in zip(test_queries, results_with_scores):
        print(f"Query: {query}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['api_id']} (score: {result['score']:.4f})")
    
    # 示例：使用重排功能
    print("\n" + "=" * 60)
    print("重排模式示例（需要启用重排器）")
    print("=" * 60)
    print("使用方法：")
    print("  retriever = create_retriever(base_dir, enable_reranker=True)")
    print("  results = retriever.search(queries, level='L2')")
    print("  # L2模式下，融合检索提供top10，重排后返回top3")
    print("=" * 60)
