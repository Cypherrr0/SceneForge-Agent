"""
代码生成和执行工具
1. 使用OpenAI生成Blender Python脚本
2. 在Blender环境中执行生成的脚本
3. API查询和验证
"""
import os
import subprocess
import tempfile
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from pathlib import Path

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


class LLMBlenderAPIQueryTool(LLMTool):
    """使用LLM从用户描述中提取Blender API查询对的工具"""
    
    def __init__(self, qwen_api_key: str, model: str = "qwen-plus"):
        """
        初始化API查询工具
        
        Args:
            api_key: OpenAI兼容API的密钥
            model: 使用的模型名称
        """
        self.qwen_api_key = qwen_api_key
        self.model = model
        self.client = OpenAI(
            api_key=self.qwen_api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="query_blender_api",
            description="Extract Queries:API pairs from user description for Blender Python operations",
            parameters=[
                ToolParameter(
                    name="user_description",
                    type="string",
                    description="User's description of the Blender operations they want to perform"
                ),
                ToolParameter(
                    name="max_tokens",
                    type="integer",
                    description="Maximum tokens for LLM response",
                    required=False,
                    default=1000
                )
            ],
            returns="Dict with success status and list of Query:API pairs",
            category="api_query"
        )
    
    def execute(self, user_description: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        从用户描述中提取Queries:API对
        
        Args:
            user_description: 用户对Blender操作的描述
            max_tokens: LLM响应的最大token数
            
        Returns:
            Dict: 包含提取的API查询对的结果
        """
        try:
            # 从外部文件读取system_prompt
            prompt_file = Path(__file__).parent / "system_prompt" / "llm_api_query"
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
            except FileNotFoundError:
                return {
                    "success": False,
                    "error": f"Prompt file not found: {prompt_file}",
                    "api_pairs": [],
                    "message": "Failed to load system prompt"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read prompt file: {e}",
                    "api_pairs": [],
                    "message": "Failed to load system prompt"
                }

            user_prompt = f"用户描述：{user_description}\n\n请提取Queries:API对："
            
            # 调用LLM
            print(f"Querying LLM for Blender API extraction...")
            print(f"User description: {user_description[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # 较低温度以获得更准确的API名称
            )
            
            # 解析响应
            llm_response = response.choices[0].message.content.strip()
            print(f"\nLLM Response:\n{llm_response}\n")
            
            # 解析Queries:API对
            api_pairs = self._parse_api_pairs(llm_response)
            
            if not api_pairs:
                return {
                    "success": False,
                    "error": "No valid API pairs found in LLM response",
                    "raw_response": llm_response,
                    "api_pairs": []
                }
            
            return {
                "success": True,
                "api_pairs": api_pairs,
                "raw_response": llm_response,
                "total_pairs": len(api_pairs),
                "message": f"Successfully extracted {len(api_pairs)} API pair(s)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "api_pairs": [],
                "message": "Failed to query Blender API"
            }
    
    def _parse_api_pairs(self, llm_response: str) -> list:
        """
        解析LLM返回的Queries:API对
        
        Args:
            llm_response: LLM的原始响应
            
        Returns:
            List[Dict]: 包含query和api的字典列表
        """
        api_pairs = []
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 查找 ":" 分隔符
            if ':' in line:
                # 分割query和api
                parts = line.split(':', 1)
                if len(parts) == 2:
                    query = parts[0].strip()
                    api = parts[1].strip()
                    
                    # 验证API格式（应该以bpy.开头）
                    if api.startswith('bpy.'):
                        api_pairs.append({
                            "query": query,
                            "api": api,
                            "description": f"{query} using {api}"
                        })
                        print(f"Parsed: {query} -> {api}")
                    else:
                        print(f"nvalid API format (not starting with 'bpy.'): {api}")
                else:
                    print(f"Invalid format: {line}")
        
        return api_pairs


class LLMBlenderAPIRetrieverL1Tool(LLMTool):
    """使用L1模式验证LLM提取的Blender API是否正确的工具"""
    
    def __init__(self, retriever_base_dir: str = None):
        """
        初始化API验证工具
        
        Args:
            retriever_base_dir: 检索器基础目录路径，如果为None则使用默认路径
        """
        self.retriever_base_dir = retriever_base_dir
        self.retriever = None
        self._retriever_initialized = False
    
    def _initialize_retriever(self):
        """延迟初始化检索器"""
        if not self._retriever_initialized:
            if self.retriever_base_dir is None:
                # 默认路径：项目根目录/rag/retrieval
                project_root = Path(__file__).parent.parent
                self.retriever_base_dir = str(project_root / "rag" / "retrieval")
            
            print(f"初始化混合检索器，基础目录: {self.retriever_base_dir}")
            
            # 导入并创建检索器
            from rag.retrieval.bm25_vector import create_retriever
            self.retriever = create_retriever(self.retriever_base_dir)
            self._retriever_initialized = True
            print("检索器初始化完成\n")
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="validate_blender_api_l1",
            description="Validate LLM-extracted Blender API pairs using L1 retrieval mode",
            parameters=[
                ToolParameter(
                    name="api_pairs",
                    type="array",
                    description="List of Query:API pairs extracted from LLMBlenderAPIQueryTool"
                ),
                ToolParameter(
                    name="use_query_expansion",
                    type="boolean",
                    description="Whether to use query expansion in retrieval",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="fusion_k",
                    type="integer",
                    description="Fusion parameter k for reciprocal rank fusion",
                    required=False,
                    default=60
                )
            ],
            returns="Dict with validated API pairs and validation statistics",
            category="api_validation"
        )
    
    def execute(self, api_pairs: List[Dict], use_query_expansion: bool = True, 
                fusion_k: int = 60) -> Dict[str, Any]:
        """
        验证LLM提取的API是否正确
        
        Args:
            api_pairs: LLMBlenderAPIQueryTool返回的api_pairs列表
            use_query_expansion: 是否使用查询扩展
            fusion_k: 融合参数k
            
        Returns:
            Dict: 包含验证结果的字典
        """
        try:
            # 初始化检索器
            if not self._retriever_initialized:
                self._initialize_retriever()
            
            if not api_pairs:
                return {
                    "success": False,
                    "error": "No API pairs provided",
                    "validated_pairs": [],
                    "statistics": {
                        "total": 0,
                        "valid": 0,
                        "invalid": 0,
                        "accuracy": 0.0
                    }
                }
            
            print(f"\n开始验证 {len(api_pairs)} 个 API 对...")
            
            # 提取所有的API作为查询
            api_queries = [pair['api'] for pair in api_pairs]
            
            print(f"提取的API查询:")
            for i, api in enumerate(api_queries, 1):
                print(f"  {i}. {api}")
            
            # 使用L1模式进行检索（返回top1）
            print(f"\n使用L1模式进行检索验证...")
            retrieval_results = self.retriever.search(
                api_queries, 
                level="L1",
                use_query_expansion=use_query_expansion,
                fusion_k=fusion_k
            )
            
            # 对比并标记
            validated_pairs = []
            valid_count = 0
            
            print(f"\n验证结果:")
            print("-" * 80)
            
            for i, (pair, retrieved_apis) in enumerate(zip(api_pairs, retrieval_results), 1):
                original_api = pair['api']
                retrieved_api = retrieved_apis[0] if retrieved_apis else None
                
                # 检查是否完全相符
                is_valid = (original_api == retrieved_api)
                
                if is_valid:
                    valid_count += 1
                    status = "VALID"
                else:
                    status = "INVALID"
                
                validated_pair = {
                    "query": pair['query'],
                    "api": original_api,
                    "retrieved_api": retrieved_api,
                    "is_valid": is_valid,
                    "description": pair.get('description', '')
                }
                
                validated_pairs.append(validated_pair)
                
                # 打印验证详情
                print(f"{i}. {status}")
                print(f"   Query: {pair['query']}")
                print(f"   Original API:  {original_api}")
                print(f"   Retrieved API: {retrieved_api}")
                if not is_valid:
                    print(f"   不匹配")
                print()
            
            # 计算统计信息
            total_count = len(api_pairs)
            invalid_count = total_count - valid_count
            accuracy = (valid_count / total_count * 100) if total_count > 0 else 0.0
            
            statistics = {
                "total": total_count,
                "valid": valid_count,
                "invalid": invalid_count,
                "accuracy": accuracy
            }
            
            print("-" * 80)
            print(f"验证统计:")
            print(f"  总数: {total_count}")
            print(f"  有效: {valid_count}")
            print(f"  无效: {invalid_count}")
            print(f"  准确率: {accuracy:.2f}%")
            print("-" * 80)
            
            return {
                "success": True,
                "validated_pairs": validated_pairs,
                "statistics": statistics,
                "message": f"Validated {total_count} API pairs with {accuracy:.2f}% accuracy"
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"验证过程出错: {e}")
            print(error_trace)
            
            return {
                "success": False,
                "error": str(e),
                "error_trace": error_trace,
                "validated_pairs": [],
                "statistics": {
                    "total": len(api_pairs) if api_pairs else 0,
                    "valid": 0,
                    "invalid": 0,
                    "accuracy": 0.0
                },
                "message": "API validation failed"
            }


class LLMBlenderAPIRetrieverL2Tool(LLMTool):
    """使用L2模式和查询扩展对验证失败的API进行二次检索的工具"""
    
    def __init__(self, qwen_api_key: str, retriever_base_dir: str = None, model: str = "qwen-plus"):
        """
        初始化API二次检索工具
        
        Args:
            qwen_api_key: OpenAI兼容API的密钥
            retriever_base_dir: 检索器基础目录路径，如果为None则使用默认路径
            model: 使用的模型名称
        """
        self.qwen_api_key = qwen_api_key
        self.model = model
        self.client = OpenAI(
            api_key=self.qwen_api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.retriever_base_dir = retriever_base_dir
        self.retriever = None
        self._retriever_initialized = False
    
    def _initialize_retriever(self):
        """延迟初始化检索器"""
        if not self._retriever_initialized:
            if self.retriever_base_dir is None:
                # 默认路径：项目根目录/rag/retrieval
                project_root = Path(__file__).parent.parent
                self.retriever_base_dir = str(project_root / "rag" / "retrieval")
            
            print(f"初始化混合检索器，基础目录: {self.retriever_base_dir}")
            
            # 导入并创建检索器
            from rag.retrieval.bm25_vector import create_retriever
            self.retriever = create_retriever(self.retriever_base_dir, enable_reranker=True)
            self._retriever_initialized = True
            print("检索器初始化完成\n")
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="retrieve_blender_api_l2",
            description="Re-retrieve APIs for failed validations using query expansion and L2 mode",
            parameters=[
                ToolParameter(
                    name="validated_pairs",
                    type="array",
                    description="List of validated Query:API pairs from LLMBlenderAPIRetrieverL1Tool"
                ),
                ToolParameter(
                    name="max_tokens",
                    type="integer",
                    description="Maximum tokens for LLM response",
                    required=False,
                    default=500
                ),
                ToolParameter(
                    name="use_query_expansion",
                    type="boolean",
                    description="Whether to use query expansion in retrieval",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="fusion_k",
                    type="integer",
                    description="Fusion parameter k for reciprocal rank fusion",
                    required=False,
                    default=60
                )
            ],
            returns="Dict with re-retrieved API candidates for failed validations",
            category="api_retrieval"
        )
    
    def execute(self, validated_pairs: List[Dict], max_tokens: int = 500,
                use_query_expansion: bool = True, fusion_k: int = 60) -> Dict[str, Any]:
        """
        对验证失败的API对进行查询扩展和二次检索
        
        Args:
            validated_pairs: LLMBlenderAPIRetrieverL1Tool返回的validated_pairs列表
            max_tokens: LLM响应的最大token数
            use_query_expansion: 是否使用查询扩展
            fusion_k: 融合参数k
            
        Returns:
            Dict: 包含重新检索结果的字典
        """
        try:
            # 初始化检索器
            if not self._retriever_initialized:
                self._initialize_retriever()
            
            if not validated_pairs:
                return {
                    "success": False,
                    "error": "No validated pairs provided",
                    "re_retrieved_pairs": []
                }
            
            # 过滤出标签为 False 的对
            failed_pairs = [pair for pair in validated_pairs if not pair.get('is_valid', True)]
            
            if not failed_pairs:
                print("所有API对验证通过，无需进行二次检索")
                return {
                    "success": True,
                    "re_retrieved_pairs": [],
                    "message": "All API pairs validated successfully, no re-retrieval needed"
                }
            
            print(f"\n发现 {len(failed_pairs)} 个验证失败的 API 对，开始二次检索...")
            
            re_retrieved_pairs = []
            
            for i, pair in enumerate(failed_pairs, 1):
                original_query = pair.get('query', '')
                original_api = pair.get('api', '')
                
                print(f"\n处理第 {i}/{len(failed_pairs)} 个失败的 API 对:")
                print(f"  原始查询: {original_query}")
                print(f"  原始API: {original_api}")
                
                # 步骤1: 使用LLM进行查询扩展
                expanded_queries = self._expand_query_with_llm(original_query, max_tokens)
                
                if not expanded_queries:
                    print(f"  查询扩展失败，跳过")
                    continue
                
                # 聚合查询：原始查询 + 扩展查询
                all_queries = [original_query] + expanded_queries
                
                print(f"  扩展后的查询列表:")
                for j, q in enumerate(all_queries, 1):
                    print(f"    {j}. {q}")
                
                # 步骤2: 对所有查询使用L2模式检索（每个查询返回top3）
                print(f"  使用L2模式进行检索...")
                retrieval_results = self.retriever.search_with_scores(
                    all_queries,
                    level="L2",  # 每个查询返回3个结果
                    use_query_expansion=use_query_expansion,
                    fusion_k=fusion_k
                )
                
                # 步骤3: 聚合所有查询的结果，去重并按分数排序
                all_candidates = {}  # api_id -> max_score
                
                for query_idx, results in enumerate(retrieval_results):
                    for result in results:
                        api_id = result['api_id']
                        # 兼容重排和非重排结果：优先使用 rerank_score，否则使用 score
                        score = result.get('rerank_score', result.get('score', 0.0))
                        # 保留每个API的最高分数
                        if api_id not in all_candidates or score > all_candidates[api_id]:
                            all_candidates[api_id] = score
                
                # 按分数排序，取top5
                sorted_candidates = sorted(
                    all_candidates.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                top5_apis = [
                    {
                        'api_id': api_id,
                        'score': score
                    }
                    for api_id, score in sorted_candidates
                ]
                
                print(f"  Top 5 候选API:")
                for j, candidate in enumerate(top5_apis, 1):
                    print(f"    {j}. {candidate['api_id']} (score: {candidate['score']:.4f})")
                
                # 构建结果
                re_retrieved_pair = {
                    "original_query": original_query,
                    "original_api": original_api,
                    "flag": False,
                    "queries": all_queries,
                    "top5_candidates": top5_apis
                }
                
                re_retrieved_pairs.append(re_retrieved_pair)
            
            print(f"\n二次检索完成，共处理 {len(re_retrieved_pairs)} 个失败的 API 对")
            
            return {
                "success": True,
                "re_retrieved_pairs": re_retrieved_pairs,
                "total_failed": len(failed_pairs),
                "total_re_retrieved": len(re_retrieved_pairs),
                "message": f"Re-retrieved {len(re_retrieved_pairs)} failed API pairs"
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"二次检索过程出错: {e}")
            print(error_trace)
            
            return {
                "success": False,
                "error": str(e),
                "error_trace": error_trace,
                "re_retrieved_pairs": [],
                "message": "API re-retrieval failed"
            }
    
    def _expand_query_with_llm(self, original_query: str, max_tokens: int = 500) -> List[str]:
        """
        使用LLM对查询语句进行同义扩展
        
        Args:
            original_query: 原始查询语句
            max_tokens: LLM响应的最大token数
            
        Returns:
            List[str]: 扩展后的查询列表（2条）
        """
        try:
            # 从外部文件读取system_prompt
            prompt_file = Path(__file__).parent / "system_prompt" / "llm_query_expand"
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
            except FileNotFoundError:
                print(f"Prompt file not found: {prompt_file}")
                return []
            except Exception as e:
                print(f"Failed to read prompt file: {e}")
                return []

            user_prompt = f"原始查询：{original_query}\n\n请提供两条同义扩展查询："
            
            print(f"    正在生成同义查询...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7  # 稍高温度以获得多样性
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # 解析响应，提取两条查询
            expanded_queries = self._parse_expanded_queries(llm_response)
            
            if len(expanded_queries) >= 2:
                print(f"    成功生成 {len(expanded_queries)} 条扩展查询")
                return expanded_queries[:2]  # 只取前两条
            else:
                print(f"    只生成了 {len(expanded_queries)} 条扩展查询")
                return expanded_queries
            
        except Exception as e:
            print(f"    LLM查询扩展失败: {e}")
            return []
    
    def _parse_expanded_queries(self, llm_response: str) -> List[str]:
        """
        解析LLM返回的扩展查询
        
        Args:
            llm_response: LLM的原始响应
            
        Returns:
            List[str]: 解析后的查询列表
        """
        queries = []
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 移除可能的编号、标点等
            # 例如: "1. add material" -> "add material"
            line = line.lstrip('0123456789.、-*• ')
            line = line.lower()
            
            # 验证：应该是英文单词，且不超过7个单词
            words = line.split()
            if 2 <= len(words) <= 7 and all(word.replace('_', '').isalnum() for word in words):
                queries.append(line)
        
        return queries


class LLMBlenderScriptExecutorTool(LLMTool):
    """在Blender环境中执行Python脚本的工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="execute_blender_script",
            description="Execute a Python script in Blender environment",
            parameters=[
                ToolParameter(
                    name="script_path",
                    type="string",
                    description="Path to the Python script to execute"
                ),
                ToolParameter(
                    name="blender_executable",
                    type="string",
                    description="Path to Blender executable",
                    required=False,
                    default="blender"
                ),
                ToolParameter(
                    name="background_mode",
                    type="boolean",
                    description="Run Blender in background mode",
                    required=False,
                    default=True
                )
            ],
            returns="Dict with execution result and output",
            category="script_execution"
        )
    
    def execute(self, script_path: str, blender_executable: str = "blender", 
                background_mode: bool = True) -> Dict[str, Any]:
        """在Blender环境中执行Python脚本"""
        try:
            # 检查脚本文件是否存在
            if not os.path.exists(script_path):
                return {
                    "success": False,
                    "error": f"Script file not found: {script_path}",
                    "message": "Script execution failed"
                }
            
            # 构建Blender命令
            if background_mode:
                cmd = [
                    blender_executable,
                    "--background",
                    "--python", script_path
                ]
            else:
                cmd = [
                    blender_executable,
                    "--python", script_path
                ]
            
            # 执行脚本
            print(f"Executing Blender script: {script_path}")
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 检查执行结果
            success = result.returncode == 0
            
            return {
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "script_path": script_path,
                "message": "Script executed successfully" if success else "Script execution failed"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timeout (5 minutes)",
                "message": "Script execution failed due to timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Script execution failed"
            }


class LLMBlenderAPIIntegratedTool(LLMTool):
    """整合API查询、验证、检索、文档增强和脚本生成的完整工具链"""
    
    def __init__(self, qwen_api_key: str, retriever_base_dir: str = None,
                 query_model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                 expansion_model: str = "Qwen/Qwen2.5-72B-Instruct",
                 script_generation_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        初始化整合工具
        
        Args:
            qwen_api_key: Qwen API密钥
            retriever_base_dir: 检索器基础目录路径
            query_model: API查询提取使用的模型
            expansion_model: 查询扩展使用的模型
            script_generation_model: 脚本生成使用的模型
        """
        self.qwen_api_key = qwen_api_key
        self.retriever_base_dir = retriever_base_dir or str(Path(__file__).parent.parent / "rag" / "retrieval")
        
        # 创建子工具实例
        self.api_query_tool = LLMBlenderAPIQueryTool(qwen_api_key, model=query_model)
        self.api_validator_tool = LLMBlenderAPIRetrieverL1Tool(retriever_base_dir)
        self.api_retriever_l2_tool = LLMBlenderAPIRetrieverL2Tool(
            qwen_api_key, retriever_base_dir, model=expansion_model
        )
        self.api_augmentation_tool = LLMBlenderAPIAugmentationTool(retriever_base_dir)
        self.script_generator_tool = LLMBlenderScriptGeneratorTool(
            qwen_api_key, model=script_generation_model
        )
        
        # 加载API元数据用于获取完整文档
        self.metadata_cache = None
        self._load_metadata()
        
        # 确保logs目录存在
        self.logs_dir = Path(__file__).parent.parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def _load_metadata(self):
        """加载API元数据"""
        try:
            metadata_file = Path(self.retriever_base_dir) / "bm25_index" / "bm25_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # 创建api_id到metadata的映射
                    self.metadata_cache = {
                        item['api_id']: item['metadata']
                        for item in data['api_mappings']
                    }
                print(f"已加载 {len(self.metadata_cache)} 个API的元数据")
            else:
                print(f"警告: 元数据文件不存在: {metadata_file}")
                self.metadata_cache = {}
        except Exception as e:
            print(f"加载元数据失败: {e}")
            self.metadata_cache = {}
    
    def _get_api_doc(self, api_id: str) -> Dict[str, Any]:
        """获取API的完整文档"""
        if api_id in self.metadata_cache:
            return self.metadata_cache[api_id]
        return {
            "api_id": api_id,
            "error": "文档未找到"
        }
    
    def _save_full_log(self, result: Dict[str, Any], user_prompt: str) -> str:
        """保存完整日志到文件"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"api_query_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "result": result
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return str(log_file)
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="generate_blender_script_with_rag",
            description="Integrated tool chain: query->validate->retrieve Blender APIs, augment documentation, and generate complete Python script",
            parameters=[
                ToolParameter(
                    name="user_prompt",
                    type="string",
                    description="User's description of Blender operations"
                ),
                ToolParameter(
                    name="top_k_l2",
                    type="integer",
                    description="Number of top candidates to return for L2 retrieval",
                    required=False,
                    default=3
                ),
                ToolParameter(
                    name="generate_script",
                    type="boolean",
                    description="Whether to generate Blender Python script",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="glb_input_path",
                    type="string",
                    description="Path to input GLB file for script generation",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="output_path",
                    type="string",
                    description="Path to output rendered image for script generation",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="script_output_dir",
                    type="string",
                    description="Directory to save the generated Blender script (default: generated_scripts/)",
                    required=False,
                    default=""
                )
            ],
            returns="Dict with API list, formatted docs, and optionally generated script",
            category="integrated_api_query"
        )
    
    def execute(self, user_prompt: str, top_k_l2: int = 3, generate_script: bool = True,
                glb_input_path: str = "", output_path: str = "", script_output_dir: str = "") -> Dict[str, Any]:
        """
        执行整合的API查询、验证、检索、文档增强和脚本生成流程
        
        Args:
            user_prompt: 用户对Blender操作的描述
            top_k_l2: L2检索返回的候选数量（默认3）
            generate_script: 是否生成Blender Python脚本（默认True）
            glb_input_path: GLB输入文件路径（可选）
            output_path: 渲染输出路径（可选）
            script_output_dir: 脚本保存目录（可选，默认为 generated_scripts/）
            
        Returns:
            Dict: 包含API列表、格式化文档和可选的生成脚本的结果
        """
        try:
            total_steps = 5 if generate_script else 4
            print("\n" + "="*80)
            print(f"开始完整的API查询和脚本生成流程 (共{total_steps}步)")
            print("="*80)
            
            # 步骤1: 提取API
            print(f"\n[1/{total_steps}] 提取API...")
            query_result = self.api_query_tool.execute(user_description=user_prompt)
            
            if not query_result['success']:
                return {
                    "success": False,
                    "error": query_result.get('error', '提取API失败'),
                    "api_list": [],
                    "formatted_docs": "",
                    "script_path": ""
                }
            
            print(f"提取到 {query_result['total_pairs']} 个API")
            
            # 步骤2: L1验证
            print(f"\n[2/{total_steps}] L1验证...")
            validation_result = self.api_validator_tool.execute(
                api_pairs=query_result['api_pairs']
            )
            
            if not validation_result['success']:
                return {
                    "success": False,
                    "error": validation_result.get('error', 'L1验证失败'),
                    "api_list": [],
                    "formatted_docs": "",
                    "script_path": ""
                }
            
            stats = validation_result['statistics']
            print(f"验证结果: {stats['valid']}/{stats['total']} 正确 ({stats['accuracy']:.1f}%)")
            
            # 步骤3: L2检索失败的API
            print(f"\n[3/{total_steps}] L2检索...")
            l2_result = self.api_retriever_l2_tool.execute(
                validated_pairs=validation_result['validated_pairs']
            )
            
            if not l2_result['success']:
                print(f"L2检索出现错误，继续处理...")
            
            # 构建最终API列表
            api_list = []
            
            for vp in validation_result['validated_pairs']:
                api_entry = {
                    "query": vp['query'],
                    "original_api": vp['api'],
                    "is_valid": vp['is_valid'],
                    "source": "L1" if vp['is_valid'] else "L2"
                }
                
                if vp['is_valid']:
                    # L1验证通过，返回对应文档
                    api_entry['api_id'] = vp['api']
                    api_entry['documentation'] = self._get_api_doc(vp['api'])
                else:
                    # L1验证失败，从L2结果中获取候选
                    candidates = []
                    
                    # 查找该API在L2结果中的候选
                    if l2_result['success'] and l2_result.get('re_retrieved_pairs'):
                        for rr_pair in l2_result['re_retrieved_pairs']:
                            if rr_pair['original_api'] == vp['api']:
                                # 取top_k_l2个候选
                                for candidate in rr_pair['top5_candidates'][:top_k_l2]:
                                    candidates.append({
                                        "api_id": candidate['api_id'],
                                        "score": candidate['score'],
                                        "documentation": self._get_api_doc(candidate['api_id'])
                                    })
                                break
                    
                    api_entry['candidates'] = candidates
                
                api_list.append(api_entry)
            
            # 步骤4: 格式化API文档
            print(f"\n[4/{total_steps}] 格式化API文档...")
            augmentation_result = self.api_augmentation_tool.execute(api_list=api_list)
            
            if not augmentation_result['success']:
                print(f"文档格式化失败: {augmentation_result.get('error', '未知错误')}")
                formatted_docs = ""
            else:
                formatted_docs = augmentation_result['formatted_docs']
            
            # 步骤5: 生成Blender脚本（可选）
            script_path = ""
            script_content = ""
            
            if generate_script:
                print(f"\n[5/{total_steps}] 生成Blender Python脚本...")
                script_result = self.script_generator_tool.execute(
                    user_description=user_prompt,
                    api_documentation=formatted_docs,
                    glb_input_path=glb_input_path,
                    output_path=output_path,
                    script_output_dir=script_output_dir
                )
                
                if script_result['success']:
                    script_path = script_result['script_path']
                    script_content = script_result['script_content']
                else:
                    print(f"脚本生成失败: {script_result.get('error', '未知错误')}")
            
            # 构建结果
            result = {
                "success": True,
                "summary": {
                    "total_apis": len(api_list),
                    "valid_apis": stats['valid'],
                    "invalid_apis": stats['invalid'],
                    "accuracy": stats['accuracy']
                },
                "api_list": api_list,
                "formatted_docs": formatted_docs,
                "script_path": script_path,
                "script_content": script_content
            }
            
            # 保存完整日志
            log_file = self._save_full_log(result, user_prompt)
            result['log_file'] = log_file
            
            # 打印简化摘要
            print("\n" + "="*80)
            print("完整流程执行完成")
            print("="*80)
            print(f"总计: {result['summary']['total_apis']} 个API")
            print(f"L1验证通过: {result['summary']['valid_apis']} 个")
            print(f"L2检索: {result['summary']['invalid_apis']} 个")
            print(f"准确率: {result['summary']['accuracy']:.1f}%")
            if generate_script and script_path:
                print(f"生成脚本路径: {script_path}")
            print(f"完整日志已保存: {log_file}")
            print("="*80 + "\n")
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"整合工具执行失败: {e}")
            print(error_trace)
            
            return {
                "success": False,
                "error": str(e),
                "error_trace": error_trace,
                "api_list": [],
                "formatted_docs": "",
                "script_path": ""
            }


class LLMBlenderAPIAugmentationTool(LLMTool):
    """API文档增强工具，将检索结果格式化为结构化的API文档"""
    
    def __init__(self, retriever_base_dir: str = None):
        """
        初始化API文档增强工具
        
        Args:
            retriever_base_dir: 检索器基础目录路径，用于加载API元数据
        """
        self.retriever_base_dir = retriever_base_dir or str(Path(__file__).parent.parent / "rag" / "retrieval")
        self.metadata_cache = None
        self._load_metadata()
    
    def _load_metadata(self):
        """加载API元数据"""
        try:
            metadata_file = Path(self.retriever_base_dir) / "bm25_index" / "bm25_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # 创建api_id到metadata的映射
                    self.metadata_cache = {
                        item['api_id']: item['metadata']
                        for item in data['api_mappings']
                    }
                print(f"已加载 {len(self.metadata_cache)} 个API的元数据")
            else:
                print(f"警告: 元数据文件不存在: {metadata_file}")
                self.metadata_cache = {}
        except Exception as e:
            print(f"加载元数据失败: {e}")
            self.metadata_cache = {}
    
    def _get_api_doc(self, api_id: str) -> Dict[str, Any]:
        """获取API的完整文档"""
        if api_id in self.metadata_cache:
            return self.metadata_cache[api_id]
        return {
            "api_id": api_id,
            "description": "文档未找到",
            "usage": "",
            "parameters": ""
        }
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="augment_blender_api_docs",
            description="Format and augment API documentation from retrieval results",
            parameters=[
                ToolParameter(
                    name="api_list",
                    type="array",
                    description="API list from LLMBlenderAPIIntegratedTool"
                )
            ],
            returns="Dict with formatted API documentation string",
            category="api_augmentation"
        )
    
    def execute(self, api_list: List[Dict]) -> Dict[str, Any]:
        """
        将API列表格式化为结构化文档
        
        Args:
            api_list: LLMBlenderAPIIntegratedTool返回的api_list
            
        Returns:
            Dict: 包含格式化后的API文档字符串
        """
        try:
            if not api_list:
                return {
                    "success": False,
                    "error": "No API list provided",
                    "formatted_docs": ""
                }
            
            print("\n" + "="*80)
            print("开始格式化API文档")
            print("="*80)
            
            formatted_sections = []
            verified_count = 0
            candidate_count = 0
            
            for item in api_list:
                query = item.get('query', '')
                is_valid = item.get('is_valid', False)
                source = item.get('source', '')
                
                if is_valid and source == "L1":
                    # L1验证通过的API
                    verified_count += 1
                    api_id = item.get('api_id', '')
                    doc = item.get('documentation', {})
                    
                    section = f"[VERIFIED] {query}\n"
                    section += f"API: {api_id}\n"
                    section += f"说明: {doc.get('description', '无说明')}\n"
                    section += f"用法: {doc.get('usage', '无用法说明')}\n"
                    section += f"参数: {doc.get('parameters', '无参数说明')}\n"
                    section += "---\n"
                    
                    formatted_sections.append(section)
                    print(f"[VERIFIED] {query} -> {api_id}")
                    
                elif not is_valid and source == "L2":
                    # L2候选API
                    candidate_count += 1
                    original_api = item.get('original_api', '')
                    candidates = item.get('candidates', [])
                    
                    section = f"[CANDIDATE] {query}\n"
                    section += f"原始API: {original_api} (验证失败)\n\n"
                    
                    for i, candidate in enumerate(candidates[:3], 1):
                        api_id = candidate.get('api_id', '')
                        score = candidate.get('score', 0.0)
                        doc = candidate.get('documentation', {})
                        
                        section += f"候选{i}: {api_id} (score: {score:.4f})\n"
                        section += f"说明: {doc.get('description', '无说明')}\n"
                        section += f"用法: {doc.get('usage', '无用法说明')}\n"
                        section += f"参数: {doc.get('parameters', '无参数说明')}\n\n"
                    
                    section += f'请根据查询"{query}"谨慎选择最合适的候选API使用\n'
                    section += "---\n"
                    
                    formatted_sections.append(section)
                    print(f"[CANDIDATE] {query} -> {len(candidates)} 个候选")
            
            # 合并所有部分
            formatted_docs = "\n".join(formatted_sections)
            
            print("\n" + "="*80)
            print("API文档格式化完成")
            print("="*80)
            print(f"已验证API: {verified_count} 个")
            print(f"候选API: {candidate_count} 个")
            print(f"总计: {len(api_list)} 个API")
            print("="*80 + "\n")
            
            return {
                "success": True,
                "formatted_docs": formatted_docs,
                "statistics": {
                    "total": len(api_list),
                    "verified": verified_count,
                    "candidates": candidate_count
                },
                "message": f"Successfully formatted {len(api_list)} API documentation entries"
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"API文档格式化失败: {e}")
            print(error_trace)
            
            return {
                "success": False,
                "error": str(e),
                "error_trace": error_trace,
                "formatted_docs": ""
            }


class LLMBlenderScriptGeneratorTool(LLMTool):
    """使用LLM和增强的API文档生成Blender Python脚本的工具"""
    
    def __init__(self, qwen_api_key: str, model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        初始化脚本生成工具
        
        Args:
            qwen_api_key: Qwen API密钥
            model: 使用的模型名称（默认使用Coder模型）
        """
        self.qwen_api_key = qwen_api_key
        self.model = model
        self.client = OpenAI(
            api_key=self.qwen_api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        
        # 确保scripts目录存在
        self.scripts_dir = Path(__file__).parent.parent / "generated_scripts"
        self.scripts_dir.mkdir(exist_ok=True)
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="generate_blender_script",
            description="Generate Blender Python script using augmented API documentation",
            parameters=[
                ToolParameter(
                    name="user_description",
                    type="string",
                    description="User's description of the Blender operations"
                ),
                ToolParameter(
                    name="api_documentation",
                    type="string",
                    description="Formatted API documentation from LLMBlenderAPIAugmentationTool"
                ),
                ToolParameter(
                    name="glb_input_path",
                    type="string",
                    description="Path to input GLB file",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="output_path",
                    type="string",
                    description="Path to output rendered image",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="script_output_dir",
                    type="string",
                    description="Directory to save the generated script (default: generated_scripts/)",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="max_tokens",
                    type="integer",
                    description="Maximum tokens for LLM response",
                    required=False,
                    default=4000
                )
            ],
            returns="Dict with generated script path and content",
            category="script_generation"
        )
    
    def execute(self, user_description: str, api_documentation: str,
                glb_input_path: str = "", output_path: str = "",
                script_output_dir: str = "", max_tokens: int = 4000) -> Dict[str, Any]:
        """
        生成Blender Python脚本
        
        Args:
            user_description: 用户对Blender操作的描述
            api_documentation: 格式化后的API文档（来自LLMBlenderAPIAugmentationTool）
            glb_input_path: GLB输入文件路径
            output_path: 渲染输出路径
            script_output_dir: 脚本保存目录（默认为 generated_scripts/）
            max_tokens: LLM响应的最大token数
            
        Returns:
            Dict: 包含生成的脚本路径和内容
        """
        try:
            # 读取system prompt模板
            prompt_file = Path(__file__).parent / "system_prompt" / "llm_script_generation"
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt_template = f.read().strip()
            except FileNotFoundError:
                return {
                    "success": False,
                    "error": f"Prompt file not found: {prompt_file}",
                    "script_path": "",
                    "script_content": ""
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read prompt file: {e}",
                    "script_path": "",
                    "script_content": ""
                }
            
            # 填充模板
            system_prompt = system_prompt_template.format(
                user_description=user_description,
                glb_input_path=glb_input_path or "请根据需要设置GLB路径",
                output_path=output_path or "请根据需要设置输出路径",
                api_documentation=api_documentation
            )
            
            print("\n" + "="*80)
            print("开始生成Blender Python脚本")
            print("="*80)
            print(f"用户描述: {user_description[:100]}...")
            print(f"GLB路径: {glb_input_path or '未指定'}")
            print(f"输出路径: {output_path or '未指定'}")
            print(f"API文档长度: {len(api_documentation)} 字符")
            
            # 调用LLM生成脚本
            user_prompt = "请根据上述要求和API文档生成完整的Blender Python脚本。只输出代码，不要任何解释。"
            
            print("\n正在调用LLM生成脚本...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2  # 较低温度以获得更稳定的代码
            )
            
            # 获取生成的脚本
            script_content = response.choices[0].message.content.strip()
            
            # 清理可能的markdown代码块标记
            if script_content.startswith("```python"):
                script_content = script_content[len("```python"):].strip()
            if script_content.startswith("```"):
                script_content = script_content[3:].strip()
            if script_content.endswith("```"):
                script_content = script_content[:-3].strip()
            
            # 保存脚本到文件
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            script_filename = f"blender_script_{timestamp}.py"
            
            # 如果指定了 script_output_dir，使用它；否则使用默认的 scripts_dir
            if script_output_dir and script_output_dir.strip():
                save_dir = Path(script_output_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                script_path = save_dir / script_filename
            else:
                script_path = self.scripts_dir / script_filename
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            print("\n" + "="*80)
            print("脚本生成完成")
            print("="*80)
            print(f"脚本路径: {script_path}")
            print(f"脚本长度: {len(script_content)} 字符")
            print(f"脚本行数: {len(script_content.splitlines())} 行")
            print("="*80 + "\n")
            
            return {
                "success": True,
                "script_path": str(script_path),
                "script_content": script_content,
                "script_length": len(script_content),
                "script_lines": len(script_content.splitlines()),
                "message": f"Script generated successfully: {script_filename}"
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"脚本生成失败: {e}")
            print(error_trace)
            
            return {
                "success": False,
                "error": str(e),
                "error_trace": error_trace,
                "script_path": "",
                "script_content": ""
            }


def register_llm_coder_tools(registry, qwen_api_key: str, retriever_base_dir: str = None,
                              query_model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                              expansion_model: str = "Qwen/Qwen2.5-72B-Instruct",
                              script_generation_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    """
    注册所有LLM代码生成和执行工具
    
    Args:
        registry: LLM工具注册中心
        qwen_api_key: Qwen API密钥
        retriever_base_dir: 检索器基础目录路径（可选）
        query_model: API查询提取使用的模型（默认: Qwen/Qwen3-Coder-480B-A35B-Instruct）
        expansion_model: 查询扩展使用的模型（默认: Qwen/Qwen2.5-72B-Instruct）
        script_generation_model: 脚本生成使用的模型（默认: Qwen/Qwen2.5-Coder-32B-Instruct）
    """
    tools = [
        LLMBlenderAPIQueryTool(qwen_api_key, model=query_model),
        LLMBlenderAPIRetrieverL1Tool(retriever_base_dir),
        LLMBlenderAPIRetrieverL2Tool(qwen_api_key, retriever_base_dir, model=expansion_model),
        LLMBlenderScriptExecutorTool(),
        LLMBlenderAPIIntegratedTool(qwen_api_key, retriever_base_dir, query_model, expansion_model, script_generation_model),
        LLMBlenderAPIAugmentationTool(retriever_base_dir),
        LLMBlenderScriptGeneratorTool(qwen_api_key, model=script_generation_model)
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} LLM coder tools\n")
    return tools
