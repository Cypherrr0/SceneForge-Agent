"""
任务规划工具
帮助LLM制定和管理任务执行计划
"""
import json
import uuid
from typing import List, Dict, Any, Optional

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


class CreateTaskPlanTool(LLMTool):
    """创建任务计划工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="create_task_plan",
            description="Create a detailed execution plan for 3D generation tasks. Automatically uses the latest parsed intent from parse_user_intent if available.",
            parameters=[
                ToolParameter(
                    name="user_request",
                    type="string", 
                    description="User's original request for 3D generation"
                ),
                ToolParameter(
                    name="include_evaluation",
                    type="boolean",
                    description="Whether to include quality evaluation steps",
                    required=False,
                    default=False
                )
            ],
            returns="Dict with detailed execution plan and todos",
            category="planning"
        )
    
    def execute(self, user_request: str, include_evaluation: bool = False) -> Dict[str, Any]:
        """创建详细的任务执行计划"""
        plan_id = str(uuid.uuid4())[:8]
        
        # 分析用户请求，提取关键信息
        task_info = self._analyze_request(user_request)
        
        # 尝试从缓存中获取解析后的意图
        try:
            from tools.intent_parser_tools import get_latest_intent
            parsed_intent = get_latest_intent()
        except ImportError:
            parsed_intent = None
        
        # 如果有解析后的意图，使用它；否则使用默认约束
        if parsed_intent and parsed_intent.get("wants"):
            print(f"   使用LLM解析的意图: {parsed_intent.get('reason', '')}")
            print(f"   解析详情: wants={parsed_intent.get('wants', {})}")
            pure_prompt = parsed_intent.get("generation_prompt", user_request)
            constraints = self._convert_intent_to_constraints(parsed_intent)
            rendering_complexity = parsed_intent.get("rendering_complexity", {
                "needs_coder_tools": False,
                "complexity_type": "simple",
                "description": "标准渲染"
            })
            print(f"   渲染复杂度: {rendering_complexity.get('description', '标准渲染')}")
        else:
            # 提取纯粹的生成提示词（去除任务描述部分）
            pure_prompt = self._extract_generation_prompt(user_request)
            
            # 使用默认约束（未找到解析意图）
            constraints = {
                "optimize_prompt": True,
                "generate_2d": True,
                "generate_3d": True,
                "render_views": True,
                "only_2d": False,
                "skip_3d": False
            }
            rendering_complexity = {
                "needs_coder_tools": False,
                "complexity_type": "simple",
                "description": "标准渲染"
            }
            print(f"   使用默认约束（未找到解析意图）")
        
        # 根据约束动态创建步骤
        steps = self._create_steps_based_on_constraints(pure_prompt, constraints, rendering_complexity)
        
        # 如果需要评估，添加评估步骤
        if include_evaluation:
            steps.extend([
                {
                    "step_id": "step_5",
                    "name": "生成评估指标",
                    "description": "为3D模型生成质量评估指标",
                    "tool": "generate_evaluation_index",
                    "parameters": {
                        "prompt": user_request,
                        "reference_image": "{image_path}"
                    },
                    "status": "pending",
                    "depends_on": ["step_2"],
                    "outputs": ["evaluation_criteria"]
                },
                {
                    "step_id": "step_6",
                    "name": "评估3D模型质量",
                    "description": "评估生成的3D模型质量",
                    "tool": "evaluate_3d_asset",
                    "parameters": {
                        "evaluation_index": "{evaluation_criteria}",
                        "render_images": "{rendered_images}"
                    },
                    "status": "pending",
                    "depends_on": ["step_4", "step_5"],
                    "outputs": ["quality_score", "improvement_suggestions"]
                }
            ])
        
        # 创建执行计划
        plan = {
            "plan_id": plan_id,
            "user_request": user_request,
            "task_type": task_info["type"],
            "estimated_duration": f"{len(steps) * 2-3} minutes",
            "total_steps": len(steps),
            "current_step": 0,
            "status": "planned",
            "steps": steps,
            "context_data": {},  # 存储步骤间传递的数据
            "include_evaluation": include_evaluation
        }
        
        # 生成todos格式的输出
        todos = []
        for i, step in enumerate(steps):
            todos.append(f"- [ ] Step {i+1}: {step['name']}")
        
        # 生成约束说明
        constraints_desc = []
        if constraints.get("only_2d"):
            constraints_desc.append("仅生成2D图像")
        if constraints.get("skip_3d"):
            constraints_desc.append("跳过3D模型生成")
        if not constraints.get("render_views"):
            constraints_desc.append("跳过渲染步骤")
        if not constraints.get("optimize_prompt"):
            constraints_desc.append("跳过提示词优化")
        
        # 添加渲染复杂度信息
        if rendering_complexity and rendering_complexity.get("needs_coder_tools", False):
            constraints_desc.append(f"使用coder工具进行{rendering_complexity.get('description', '复杂渲染')}")
        
        constraints_info = "，".join(constraints_desc) if constraints_desc else "完整3D生成流程"
        
        return {
            "success": True,
            "plan_id": plan_id,
            "execution_plan": plan,
            "todos_list": "\n".join(todos),
            "constraints": constraints_info,
            "total_steps": len(steps),
            "next_action": "执行第一个步骤",
            "ready_to_execute": True,
            "plan_description": f"任务计划已创建，共{len(steps)}个步骤。执行完所有{len(steps)}个步骤后任务即完成，不要执行额外步骤。"
        }
    
    def _convert_intent_to_constraints(self, parsed_intent: Dict[str, Any]) -> Dict[str, Any]:
        """将LLM解析的意图转换为约束格式"""
        wants = parsed_intent.get("wants", {})
        intent_constraints = parsed_intent.get("constraints", {})
        
        constraints = {
            "optimize_prompt": wants.get("optimize_prompt", True),
            "generate_2d": wants.get("generate_2d", True),
            "generate_3d": wants.get("generate_3d", True),
            "render_views": wants.get("render_views", True),
            "only_2d": intent_constraints.get("only_2d", False),
            "skip_3d": intent_constraints.get("skip_3d", False)
        }
        
        return constraints
    
    
    def _create_steps_based_on_constraints(self, pure_prompt: str, constraints: Dict[str, Any], rendering_complexity: Dict[str, Any] = None) -> List[Dict]:
        """根据约束动态创建执行步骤"""
        steps = []
        step_counter = 1
        last_step_id = None
        
        # 【关键修改】尝试从会话memory中获取output_dir，如果存在则复用
        try:
            from tools.intent_parser_tools import get_session_memory
            session_memory = get_session_memory()
            saved_output_dir = session_memory.get("output_dir")
            if saved_output_dir:
                print(f"   从会话memory中复用output_dir: {saved_output_dir}")
        except:
            saved_output_dir = None
        
        # Step 1: 优化提示词（如果需要）
        if constraints.get("optimize_prompt", True):
            step_id = f"step_{step_counter}"
            steps.append({
                "step_id": step_id,
                "name": "优化提示词",
                "description": "优化用户提示词以获得更好的生成效果",
                "tool": "optimize_3d_prompt",
                "parameters": {
                    "original_prompt": pure_prompt,
                    "target_model": "gemini",
                    "translate_to_english": False
                },
                "status": "pending",
                "depends_on": [],
                "outputs": ["optimized_prompt"]
            })
            last_step_id = step_id
            step_counter += 1
        
        # Step 2: 生成2D图像（如果需要）
        if constraints.get("generate_2d", True):
            step_id = f"step_{step_counter}"
            depends_on = [last_step_id] if last_step_id else []
            prompt_param = "{optimized_prompt}" if constraints.get("optimize_prompt", True) else pure_prompt
            
            # 【关键修改】如果有saved_output_dir，使用固定路径；否则使用占位符
            if saved_output_dir:
                save_path_param = f"{saved_output_dir}/2d_reference_image.png"
            else:
                save_path_param = "{output_dir}/2d_reference_image.png"
            
            steps.append({
                "step_id": step_id,
                "name": "生成2D参考图像",
                "description": "使用提示词生成2D参考图像",
                "tool": "text_to_image",
                "parameters": {
                    "prompt": prompt_param,
                    "style": "3d",
                    "save_path": save_path_param
                },
                "status": "pending",
                "depends_on": depends_on,
                "outputs": ["image_path"]
            })
            last_step_id = step_id
            step_counter += 1
        
        # Step 3: 生成3D模型（如果需要）
        if constraints.get("generate_3d", True):
            step_id = f"step_{step_counter}"
            depends_on = [last_step_id] if last_step_id else []
            prompt_param = "{optimized_prompt}" if constraints.get("optimize_prompt", True) else pure_prompt
            
            steps.append({
                "step_id": step_id,
                "name": "生成3D模型",
                "description": "基于2D图像生成3D模型",
                "tool": "img_to_3d_complete",
                "parameters": {
                    "prompt": prompt_param,
                    "use_existing_image": "{image_path}",
                    "uid": "{uid}",
                    "save_dir": "{save_dir}"
                },
                "status": "pending",
                "depends_on": depends_on,
                "outputs": ["glb_path", "uid", "output_dir"]
            })
            last_step_id = step_id
            step_counter += 1
        
        # Step 4: 渲染多视角（如果需要）
        if constraints.get("render_views", True):
            step_id = f"step_{step_counter}"
            depends_on = [last_step_id] if last_step_id else []
            
            # 检查是否需要使用coder工具
            if rendering_complexity and rendering_complexity.get("needs_coder_tools", False):
                # 构建完整的Blender操作描述
                rendering_description = rendering_complexity.get('description', '复杂渲染')
                # 组合完整的操作指令：首先描述场景内容，然后描述摄像机操作
                full_rendering_prompt = f"渲染操作: {rendering_description}"
                
                # 使用集成的RAG+脚本生成工具进行复杂渲染
                steps.append({
                    "step_id": step_id,
                    "name": "生成复杂渲染脚本",
                    "description": f"使用RAG检索API并生成Blender脚本实现{rendering_description}",
                    "tool": "generate_blender_script_with_rag",
                    "parameters": {
                        "user_prompt": full_rendering_prompt,
                        "generate_script": True,
                        "glb_input_path": "{glb_path}",
                        "output_path": "{output_dir}/rendered_output.png",
                        "script_output_dir": "{output_dir}"
                    },
                    "status": "pending",
                    "depends_on": depends_on,
                    "outputs": ["script_path", "script_content", "api_list", "formatted_docs"]
                })
                step_counter += 1
                
                # 执行生成的脚本
                step_id = f"step_{step_counter}"
                steps.append({
                    "step_id": step_id,
                    "name": "执行渲染脚本",
                    "description": "在Blender中执行生成的渲染脚本",
                    "tool": "execute_blender_script",
                    "parameters": {
                        "script_path": "{script_path}",
                        "blender_executable": "blender",
                        "background_mode": True
                    },
                    "status": "pending",
                    "depends_on": [f"step_{step_counter-1}"],
                    "outputs": ["rendered_images", "execution_result"]
                })
                step_counter += 1
            else:
                # 使用标准渲染工具
                steps.append({
                    "step_id": step_id,
                    "name": "渲染多视角图像",
                    "description": "渲染3D模型的多个视角",
                    "tool": "render_3d_scene",
                    "parameters": {
                        "glb_file_path": "{glb_path}",
                        "output_dir": "{output_dir}/renders"
                    },
                    "status": "pending",
                    "depends_on": depends_on,
                    "outputs": ["rendered_images", "blend_file"]
                })
                step_counter += 1
        
        return steps
    
    def _extract_generation_prompt(self, request: str) -> str:
        """从用户请求中提取纯粹的生成提示词"""
        import re
        
        # 首先移除任务约束部分
        constraints_patterns = [
            r"[，。,\.]\s*不要生成3D.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*只生成2D.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*不需要.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*跳过.*?(?=[，。,\.\n]|$)"
        ]
        
        clean_request = request
        for pattern in constraints_patterns:
            clean_request = re.sub(pattern, "", clean_request)
        
        # 查找"生成3D模型："后面的内容
        pattern1 = r"生成3D模型[：:]\s*(.+?)(?:\n|$)"
        match1 = re.search(pattern1, clean_request)
        if match1:
            return match1.group(1).strip()
        
        # 查找"生成"、"创建"等关键词后的描述
        pattern2 = r"(?:生成|创建|制作)\s*(?:一个|一只)?\s*(.+?)(?:\n|要求|请|。|$)"
        match2 = re.search(pattern2, clean_request)
        if match2:
            return match2.group(1).strip()
        
        # 如果都没找到，寻找最像描述的部分
        lines = clean_request.split('\n')
        for line in lines:
            line = line.strip()
            if any(word in line for word in ["猫", "狗", "花瓶", "椅子", "企鹅", "北极熊", "cat", "dog", "vase", "chair"]):
                # 清理任务描述词汇
                clean_line = re.sub(r"(?:生成|创建|制作|generate|create)\s*(?:3D模型)?[：:]?\s*", "", line)
                clean_line = re.sub(r"(?:要求|请|立即|开始|执行).*", "", clean_line)
                if clean_line.strip():
                    return clean_line.strip()
        
        # 最后的后备方案
        return clean_request.strip()
    
    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """分析用户请求，提取任务信息"""
        request_lower = request.lower()
        
        # 简单的任务类型识别
        if any(word in request_lower for word in ["生成", "创建", "制作", "generate", "create", "make"]):
            task_type = "generation"
        else:
            task_type = "unknown"
        
        # 提取对象信息
        objects = []
        if "猫" in request or "cat" in request_lower:
            objects.append("cat")
        if "椅子" in request or "chair" in request_lower:
            objects.append("chair")
        
        return {
            "type": task_type,
            "objects": objects,
            "complexity": "simple" if len(objects) <= 2 else "complex"
        }


class ExecuteNextStepTool(LLMTool):
    """执行下一个计划步骤工具"""
    
    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="execute_next_step",
            description="Execute the next step in the execution plan",
            parameters=[
                ToolParameter(
                    name="execution_plan",
                    type="object",
                    description="Current execution plan with steps"
                )
            ],
            returns="Dict with updated plan and step execution result",
            category="planning"
        )
    
    def execute(self, execution_plan: Dict) -> Dict[str, Any]:
        """执行计划中的下一个步骤"""
        steps = execution_plan.get("steps", [])
        current_step_index = execution_plan.get("current_step", 0)
        context_data = execution_plan.get("context_data", {})
        
        if current_step_index >= len(steps):
            return {
                "success": True,
                "message": "所有步骤已完成",
                "execution_plan": execution_plan,
                "completed": True
            }
        
        current_step = steps[current_step_index]
        step_name = current_step["name"]
        tool_name = current_step["tool"]
        parameters = current_step["parameters"].copy()
        
        print(f"执行步骤 {current_step_index + 1}: {step_name}")
        
        # 检查依赖是否满足
        depends_on = current_step.get("depends_on", [])
        for dep_step_id in depends_on:
            dep_step = next((s for s in steps if s["step_id"] == dep_step_id), None)
            if not dep_step or dep_step.get("status") != "completed":
                return {
                    "success": False,
                    "error": f"依赖步骤 {dep_step_id} 未完成",
                    "execution_plan": execution_plan
                }
        
        # 解析参数中的占位符
        parameters = self._resolve_parameters(parameters, context_data)
        
        try:
            # 标记为进行中
            current_step["status"] = "in_progress"
            
            # 执行工具
            if self.tool_registry:
                result = self.tool_registry.execute_function_call(tool_name, parameters)
            else:
                # 模拟执行
                result = {
                    "success": True,
                    "mock": True,
                    "step": step_name
                }
            
            # 处理结果
            if isinstance(result, dict) and result.get("success"):
                current_step["status"] = "completed"
                current_step["result"] = result
                
                # 更新上下文数据
                for output_key in current_step.get("outputs", []):
                    if output_key in result:
                        context_data[output_key] = result[output_key]
                
                # 更新计划
                execution_plan["current_step"] = current_step_index + 1
                execution_plan["context_data"] = context_data
                
                print(f"✅ 步骤 {current_step_index + 1} 完成: {step_name}")
                
                return {
                    "success": True,
                    "execution_plan": execution_plan,
                    "step_result": result,
                    "completed_step": current_step_index + 1,
                    "next_step": current_step_index + 1 if current_step_index + 1 < len(steps) else None,
                    "message": f"步骤 {current_step_index + 1} 完成，{'准备执行下一步' if current_step_index + 1 < len(steps) else '所有步骤完成'}"
                }
            else:
                current_step["status"] = "failed"
                current_step["error"] = str(result)
                
                return {
                    "success": False,
                    "execution_plan": execution_plan,
                    "error": f"步骤 {current_step_index + 1} 失败: {result}",
                    "failed_step": current_step_index + 1
                }
                
        except Exception as e:
            current_step["status"] = "failed"
            current_step["error"] = str(e)
            
            return {
                "success": False,
                "execution_plan": execution_plan,
                "error": f"步骤 {current_step_index + 1} 执行异常: {str(e)}",
                "failed_step": current_step_index + 1
            }
    
    def _resolve_parameters(self, parameters: Dict, context_data: Dict) -> Dict:
        """解析参数中的占位符"""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                placeholder_key = value[1:-1]
                if placeholder_key in context_data:
                    resolved[key] = context_data[placeholder_key]
                    print(f"🔗 解析占位符 {value} -> {context_data[placeholder_key]}")
                else:
                    print(f"⚠️ 占位符 {value} 在上下文中未找到")
                    resolved[key] = value  # 保留原值
            else:
                resolved[key] = value
        
        return resolved


def register_planning_tools(registry, tool_registry=None):
    """
    注册任务规划工具
    
    Args:
        registry: LLM工具注册中心
        tool_registry: 基础工具注册中心
    """
    tools = [
        CreateTaskPlanTool(),
        # ExecuteNextStepTool 已禁用：
        # 1. 会导致JSON解析失败（execution_plan对象太大）
        # 2. LLM可以直接调用实际工具，不需要这个中间工具
        # ExecuteNextStepTool(tool_registry)
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} planning tools\n")
    return tools
