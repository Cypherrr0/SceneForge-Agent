#!/usr/bin/env python3
"""
评估BM25检索性能
计算Hit Rate和其他指标
优化版：支持同义词和查询扩展
"""
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import re


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
            print(f"已加载配置: {len(SYNONYM_DICT)}个同义词, {len(API_PATTERNS)}个API模式")
    else:
        print("未找到配置文件，使用默认分词")


def apply_synonyms(text: str) -> str:
    """应用同义词替换"""
    text_lower = text.lower()
    for original, synonym in SYNONYM_DICT.items():
        pattern = r'\b' + re.escape(original) + r'\b'
        text_lower = re.sub(pattern, synonym, text_lower)
    return text_lower


def tokenize_text(text: str, apply_synonym: bool = True) -> List[str]:
    """
    与build_bm25_index.py中相同的分词函数
    """
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
    
    # 加载BM25对象
    with open(index_path / "bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    # 加载元数据
    with open(index_path / "bm25_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return bm25, metadata


def generate_test_queries(api_doc: Dict) -> List[str]:
    """
    从API文档生成测试查询
    
    策略：
    1. 使用description作为自然语言查询
    2. 使用API名称的变体
    3. 使用description的前半部分
    4. 使用关键动词+名词组合
    """
    queries = []
    
    name = api_doc.get('name', '')
    description = api_doc.get('description', '')
    full_name = api_doc.get('full_name', '')
    
    # Query 1: 完整描述
    if description:
        queries.append(description)
    
    # Query 2: 描述的前半部分（模拟用户简短查询）
    if description:
        words = description.split()
        if len(words) > 5:
            short_desc = ' '.join(words[:5])
            queries.append(short_desc)
    
    # Query 3: API名称（下划线转空格）
    if name:
        name_as_query = name.replace('_', ' ')
        queries.append(name_as_query)
    
    # Query 4: 提取关键动词和名词
    if description:
        # 简单提取前3个有意义的词
        words = [w.lower() for w in description.split() if len(w) > 3][:3]
        if words:
            queries.append(' '.join(words))
    
    return queries


def search_bm25(bm25, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
    """
    使用BM25检索
    
    Returns:
        List of (index, score) tuples
    """
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, scores[idx]) for idx in top_indices]


def calculate_metrics(all_results: List[Dict]) -> Dict:
    """
    计算各种评估指标
    
    Metrics:
    - Hit Rate @K: 正确答案在Top-K中的比例
    - MRR (Mean Reciprocal Rank): 平均倒数排名
    - Success Rate: 至少一个查询成功的API比例
    """
    metrics = {
        'total_apis': len(all_results),
        'total_queries': sum(len(r['queries']) for r in all_results),
        'hit_at_1': 0,
        'hit_at_5': 0,
        'hit_at_10': 0,
        'mrr_sum': 0.0,
        'success_apis': 0,
        'failed_apis': []
    }
    
    for result in all_results:
        api_id = result['api_id']
        query_results = result['query_results']
        
        api_has_success = False
        
        for qr in query_results:
            rank = qr['correct_rank']
            
            if rank is not None:
                # 命中
                api_has_success = True
                
                if rank == 1:
                    metrics['hit_at_1'] += 1
                if rank <= 5:
                    metrics['hit_at_5'] += 1
                if rank <= 10:
                    metrics['hit_at_10'] += 1
                
                # MRR: 1/rank
                metrics['mrr_sum'] += 1.0 / rank
        
        if api_has_success:
            metrics['success_apis'] += 1
        else:
            metrics['failed_apis'].append(api_id)
    
    # 计算比例
    total_queries = metrics['total_queries']
    if total_queries > 0:
        metrics['hit_rate_at_1'] = metrics['hit_at_1'] / total_queries
        metrics['hit_rate_at_5'] = metrics['hit_at_5'] / total_queries
        metrics['hit_rate_at_10'] = metrics['hit_at_10'] / total_queries
        metrics['mrr'] = metrics['mrr_sum'] / total_queries
    
    metrics['api_success_rate'] = metrics['success_apis'] / metrics['total_apis']
    
    return metrics


def evaluate_bm25(
    bm25,
    metadata: Dict,
    original_docs: List[Dict],
    sample_size: int = None
) -> Tuple[Dict, List[Dict]]:
    """
    评估BM25性能
    
    Args:
        bm25: BM25索引对象
        metadata: 索引元数据
        original_docs: 原始API文档
        sample_size: 采样数量（None表示全部）
    
    Returns:
        metrics: 评估指标
        all_results: 详细结果
    """
    # 创建API ID到索引的映射
    api_id_to_index = {}
    for idx, mapping in enumerate(metadata['api_mappings']):
        api_id_to_index[mapping['api_id']] = idx
    
    # 采样或使用全部文档
    if sample_size:
        test_docs = random.sample(original_docs, min(sample_size, len(original_docs)))
        print(f"采样评估: {len(test_docs)} 个API")
    else:
        test_docs = original_docs
        print(f"全量评估: {len(test_docs)} 个API")
    
    all_results = []
    
    print("开始评估...")
    for i, api_doc in enumerate(test_docs):
        if (i + 1) % 100 == 0:
            print(f"  已评估: {i + 1}/{len(test_docs)}")
        
        api_id = api_doc.get('id') or api_doc.get('full_name')
        
        # 检查API是否在索引中
        if api_id not in api_id_to_index:
            continue
        
        correct_index = api_id_to_index[api_id]
        
        # 生成测试查询
        queries = generate_test_queries(api_doc)
        
        # 对每个查询进行检索
        query_results = []
        for query in queries:
            query_tokens = tokenize_text(query)
            
            if not query_tokens:
                continue
            
            # BM25检索
            top_results = search_bm25(bm25, query_tokens, top_k=10)
            
            # 检查正确答案的排名
            correct_rank = None
            for rank, (idx, score) in enumerate(top_results, 1):
                if idx == correct_index:
                    correct_rank = rank
                    break
            
            query_results.append({
                'query': query,
                'query_tokens': query_tokens,
                'correct_rank': correct_rank,
                'top_5': [(metadata['api_mappings'][idx]['api_id'], score) 
                          for idx, score in top_results[:5]]
            })
        
        all_results.append({
            'api_id': api_id,
            'queries': queries,
            'query_results': query_results
        })
    
    # 计算指标
    metrics = calculate_metrics(all_results)
    
    return metrics, all_results


def print_metrics(metrics: Dict):
    """打印评估指标"""
    print(f"\n{'='*60}")
    print("BM25检索性能评估")
    print(f"{'='*60}")
    print(f"总API数: {metrics['total_apis']}")
    print(f"总查询数: {metrics['total_queries']}")
    print(f"平均每API查询数: {metrics['total_queries']/metrics['total_apis']:.1f}")
    
    print(f"\n命中率 (Hit Rate):")
    print(f"  Hit@1:  {metrics['hit_rate_at_1']:.2%} ({metrics['hit_at_1']}/{metrics['total_queries']})")
    print(f"  Hit@5:  {metrics['hit_rate_at_5']:.2%} ({metrics['hit_at_5']}/{metrics['total_queries']})")
    print(f"  Hit@10: {metrics['hit_rate_at_10']:.2%} ({metrics['hit_at_10']}/{metrics['total_queries']})")
    
    print(f"\n其他指标:")
    print(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
    print(f"  API成功率: {metrics['api_success_rate']:.2%} ({metrics['success_apis']}/{metrics['total_apis']})")
    
    print(f"\n失败的API数量: {len(metrics['failed_apis'])}")


def analyze_failures(all_results: List[Dict], metrics: Dict, top_n: int = 10):
    """分析失败案例"""
    print(f"\n{'='*60}")
    print(f"失败案例分析 (显示前{top_n}个)")
    print(f"{'='*60}")
    
    failed_cases = []
    for result in all_results:
        # 检查是否所有查询都失败
        all_failed = all(qr['correct_rank'] is None for qr in result['query_results'])
        if all_failed:
            failed_cases.append(result)
    
    for i, case in enumerate(failed_cases[:top_n], 1):
        print(f"\n{i}. API: {case['api_id']}")
        print(f"   测试查询数: {len(case['queries'])}")
        
        for j, qr in enumerate(case['query_results'][:2], 1):  # 只显示前2个查询
            print(f"\n   查询{j}: {qr['query']}")
            print(f"   分词: {qr['query_tokens']}")
            print(f"   Top-3结果:")
            for k, (api_id, score) in enumerate(qr['top_5'][:3], 1):
                print(f"     {k}. [{score:.2f}] {api_id}")


def analyze_by_category(all_results: List[Dict], original_docs: List[Dict]):
    """按分类分析性能"""
    print(f"\n{'='*60}")
    print("按类别分析")
    print(f"{'='*60}")
    
    # 创建API ID到category的映射
    api_to_category = {}
    for doc in original_docs:
        api_id = doc.get('id') or doc.get('full_name')
        category = doc.get('category', 'unknown')
        api_to_category[api_id] = category
    
    # 按category统计
    category_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'queries': 0, 'hits': 0})
    
    for result in all_results:
        api_id = result['api_id']
        category = api_to_category.get(api_id, 'unknown')
        
        has_success = any(qr['correct_rank'] is not None for qr in result['query_results'])
        
        category_stats[category]['total'] += 1
        if has_success:
            category_stats[category]['success'] += 1
        
        for qr in result['query_results']:
            category_stats[category]['queries'] += 1
            if qr['correct_rank'] is not None and qr['correct_rank'] <= 5:
                category_stats[category]['hits'] += 1
    
    # 按成功率排序
    sorted_categories = sorted(category_stats.items(), 
                               key=lambda x: x[1]['success']/x[1]['total'] if x[1]['total'] > 0 else 0,
                               reverse=True)
    
    print(f"\n{'类别':<20} {'API数':<8} {'成功率':<10} {'Hit@5':<10}")
    print("-" * 60)
    
    for category, stats in sorted_categories:
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        hit_rate = stats['hits'] / stats['queries'] if stats['queries'] > 0 else 0
        print(f"{category:<20} {stats['total']:<8} {success_rate:>8.1%} {hit_rate:>10.1%}")


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    
    # 路径配置
    bm25_index_dir = base_dir / "bm25_index"
    original_docs_file = base_dir / "structured_docs" / "bpy_ops_flat.json"
    
    # 检查文件
    if not bm25_index_dir.exists():
        print(f"错误: BM25索引目录不存在: {bm25_index_dir}")
        print("请先运行 build_bm25_index.py")
        return
    
    if not original_docs_file.exists():
        print(f"错误: 原始文档不存在: {original_docs_file}")
        return
    
    print(f"{'='*60}")
    print("BM25性能评估")
    print(f"{'='*60}")
    print(f"索引目录: {bm25_index_dir}")
    print(f"原始文档: {original_docs_file}")
    print(f"{'='*60}\n")
    
    # 加载BM25配置
    config_file = bm25_index_dir / "bm25_config.json"
    load_bm25_config(str(config_file))
    
    # 加载BM25索引
    print("加载BM25索引...")
    bm25, metadata = load_bm25_index(str(bm25_index_dir))
    print(f"索引已加载: {len(metadata['api_mappings'])} 个API")
    
    # 加载原始文档
    print("加载原始文档...")
    with open(original_docs_file, 'r', encoding='utf-8') as f:
        original_docs = json.load(f)
    print(f"原始文档已加载: {len(original_docs)} 个API")
    
    # 设置随机种子
    random.seed(42)
    
    # 评估（可选：设置sample_size=100进行快速测试）
    # metrics, all_results = evaluate_bm25(bm25, metadata, original_docs, sample_size=100)
    metrics, all_results = evaluate_bm25(bm25, metadata, original_docs, sample_size=None)
    
    # 打印结果
    print_metrics(metrics)
    
    # 分析失败案例
    analyze_failures(all_results, metrics, top_n=10)
    
    # 按类别分析
    analyze_by_category(all_results, original_docs)
    
    # 保存详细结果
    output_file = base_dir / "bm25_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'sample_results': all_results[:50]  # 只保存前50个详细结果
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()

