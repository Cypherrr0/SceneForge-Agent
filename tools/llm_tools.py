"""
LLM工具调用框架
支持OpenAI Function Calling格式的工具定义和调用
"""
import json
import os
import inspect
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

from tools.base import BaseTool, ToolSchema, ToolParameter


class ParameterType(str, Enum):
    """参数类型枚举"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class FunctionParameter:
    """OpenAI Function参数定义"""
    type: str
    description: str
    enum: Optional[List[str]] = None
    items: Optional[Dict] = None
    properties: Optional[Dict] = None
    required: Optional[bool] = None


@dataclass
class FunctionDefinition:
    """OpenAI Function定义"""
    name: str
    description: str
    parameters: Dict[str, Any]


class LLMTool(BaseTool):
    """
    支持LLM调用的工具基类
    扩展BaseTool以支持OpenAI Function Calling格式
    """
    
    def to_function_definition(self) -> Dict[str, Any]:
        """
        转换为OpenAI Function Calling格式
        
        Returns:
            Dict: OpenAI格式的函数定义
        """
        properties = {}
        required = []
        
        for param in self.schema.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            
            if hasattr(param, 'enum') and param.enum:
                properties[param.name]["enum"] = param.enum
                
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def parse_arguments(self, arguments: Union[str, Dict]) -> Dict[str, Any]:
        """
        解析LLM返回的参数
        
        Args:
            arguments: LLM返回的参数（可能是JSON字符串或字典）
            
        Returns:
            Dict: 解析后的参数字典
        """
        # 处理字符串格式的参数（支持多层编码）
        parsed_arguments = arguments
        
        # 循环解析直到得到字典对象
        max_parse_attempts = 3
        for attempt in range(max_parse_attempts):
            if isinstance(parsed_arguments, str):
                try:
                    parsed_arguments = json.loads(parsed_arguments)
                    #print(f"   🔍 JSON解析尝试 {attempt + 1}，结果类型: {type(parsed_arguments)}")
                except json.JSONDecodeError as e:
                    print(f"   JSON解析失败 (尝试 {attempt + 1}): {e}")
                    raise ValueError(f"Invalid JSON arguments: {arguments[:100]}... Error: {e}")
            else:
                break  # 已经不是字符串了，停止解析
        
        # 确保最终结果是字典类型
        if not isinstance(parsed_arguments, dict):
            #print(f"   ❌ 经过 {max_parse_attempts} 次解析仍不是字典: {type(parsed_arguments)}")
            print(f"   最终内容: {str(parsed_arguments)[:200]}...")
            raise ValueError(f"Cannot parse arguments to dictionary after {max_parse_attempts} attempts")
        
        #print(f"   参数解析成功，字典包含 {len(parsed_arguments)} 个参数")
        
        # 验证参数
        try:
            validated = self.validate_parameters(**parsed_arguments)
            #print(f"   参数验证成功，验证后参数: {len(validated)} 个")
            return validated
        except Exception as e:
            print(f"   参数验证失败: {e}")
            print(f"   原始参数: {parsed_arguments}")
            raise ValueError(f"Parameter validation failed: {e}. Arguments: {parsed_arguments}")


class LLMToolRegistry:
    """
    LLM工具注册中心
    管理可供LLM调用的工具
    """
    
    def __init__(self):
        self._tools: Dict[str, LLMTool] = {}
        self._function_definitions: List[Dict] = []
    
    def register(self, tool: LLMTool, override: bool = False) -> None:
        """
        注册LLM工具
        
        Args:
            tool: LLM工具实例
            override: 是否覆盖已存在的工具
        """
        tool_name = tool.schema.name
        
        if tool_name in self._tools and not override:
            raise ValueError(f"Tool '{tool_name}' already registered")
        
        self._tools[tool_name] = tool
        
        # 更新函数定义列表
        self._update_function_definitions()
        
        print(f"Registered LLM tool: {tool_name}")
    
    def _update_function_definitions(self):
        """更新函数定义列表"""
        self._function_definitions = [
            tool.to_function_definition() 
            for tool in self._tools.values()
        ]
    
    def get_function_definitions(self) -> List[Dict]:
        """
        获取所有函数定义（用于LLM调用）
        
        Returns:
            List[Dict]: OpenAI格式的函数定义列表
        """
        return self._function_definitions
    
    def execute_function_call(self, function_name: str, 
                             arguments: Union[str, Dict]) -> Any:
        """
        执行LLM的函数调用
        
        Args:
            function_name: 函数名称
            arguments: 函数参数
            
        Returns:
            函数执行结果
        """
        if function_name not in self._tools:
            raise ValueError(f"Function '{function_name}' not found")
        
        tool = self._tools[function_name]
        
        # 打印工具调用信息
        tool_class_name = tool.__class__.__name__
        print(f"Calling tool: {tool_class_name}")
        
        
        # 调试：显示参数类型和内容
        #print(f"   📋 参数类型: {type(arguments)}")
        #if isinstance(arguments, str):
        #    print(f"   📋 参数内容: {arguments[:100]}...")
        #else:
        #    print(f"   📋 参数内容: {str(arguments)[:100]}...")
        
        # 解析参数
        try:
            parsed_args = tool.parse_arguments(arguments)
            #print(f"   参数解析成功")
        except Exception as e:
            print(f"   参数解析失败: {e}")
            raise
        
        # 执行工具
        try:
            result = tool.execute(**parsed_args)
            
            # 自动更新会话级别的持久化memory（从工具返回结果中提取关键路径）
            if isinstance(result, dict):
                try:
                    from tools.intent_parser_tools import update_session_memory_from_tool_result
                    update_session_memory_from_tool_result(result)
                except:
                    pass  # 静默失败，不影响工具执行
            
            # 显示工具执行结果
            if isinstance(result, dict):
                success = result.get("success", True)  # 默认认为成功
                if success:
                    print(f"   {tool_class_name} 执行成功")
                    # 显示关键结果信息
                    if "execution_plan" in result and "todos_list" in result:
                        # 显示任务计划
                        plan = result['execution_plan']
                        print(f"   计划ID: {result.get('plan_id', 'N/A')}")
                        print(f"   总步骤数: {plan.get('total_steps', 0)}")
                        if "constraints" in result:
                            print(f"   任务约束: {result['constraints']}")
                        if "plan_description" in result:
                            print(f"   重要提示: {result['plan_description']}")
                        print(f"\n   执行计划:")
                        print(f"   {result['todos_list']}")
                        print(f"\n   下一步行动: {result.get('next_action', '未定义')}")
                    elif "optimized_prompt" in result:
                        print(f"   优化结果: {result['optimized_prompt'][:60]}...")
                    elif "image_path" in result:
                        print(f"   图像路径: {result['image_path']}")
                    elif "glb_path" in result:
                        print(f"   3D模型: {result['glb_path']}")
                    
                    # 显示stdout和stderr（如果存在）
                    if "stdout" in result and result["stdout"]:
                        print(f"\n   标准输出:")
                        # 只显示最后50行，避免输出过长
                        stdout_lines = result["stdout"].strip().split('\n')
                        display_lines = stdout_lines[-50:] if len(stdout_lines) > 50 else stdout_lines
                        for line in display_lines:
                            print(f"      {line}")
                    
                    if "stderr" in result and result["stderr"]:
                        print(f"\n   错误输出:")
                        stderr_lines = result["stderr"].strip().split('\n')
                        display_lines = stderr_lines[-50:] if len(stderr_lines) > 50 else stderr_lines
                        for line in display_lines:
                            print(f"      {line}")
                else:
                    print(f"   {tool_class_name} 执行失败: {result.get('error', '未知错误')}")
                    # 执行失败时也显示stderr
                    if "stderr" in result and result["stderr"]:
                        print(f"\n   错误输出:")
                        stderr_lines = result["stderr"].strip().split('\n')
                        display_lines = stderr_lines[-50:] if len(stderr_lines) > 50 else stderr_lines
                        for line in display_lines:
                            print(f"      {line}")
            else:
                print(f"   {tool_class_name} 执行完成")
            
            return result
            
        except Exception as e:
            print(f"   {tool_class_name} 执行异常: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_tool_calls(self, tool_calls: List) -> List[Dict]:
        """
        批量处理工具调用
        
        Args:
            tool_calls: LLM返回的工具调用列表
            
        Returns:
            List[Dict]: 工具执行结果列表
        """
        results = []
        
        for tool_call in tool_calls:
            function_name = "unknown"
            tool_call_id = None
            
            try:
                # 处理不同格式的tool_call
                if hasattr(tool_call, 'function'):
                    # OpenAI SDK对象格式
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    tool_call_id = getattr(tool_call, 'id', None)
                elif isinstance(tool_call, dict):
                    # 字典格式
                    function_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    tool_call_id = tool_call.get("id")
                else:
                    raise ValueError(f"Unsupported tool_call format: {type(tool_call)}")
                
                # 不在这里打印，让 execute_function_call 统一处理打印
                
                result = self.execute_function_call(function_name, arguments)
                
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result) if not isinstance(result, str) else result
                })
                
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        
        return results
    
    def get_tool(self, tool_name: str) -> Optional[LLMTool]:
        """获取工具实例"""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())
    
    def get_tools_description(self) -> str:
        """
        获取所有工具的描述（用于系统提示）
        
        Returns:
            str: 工具描述文本
        """
        descriptions = []
        for tool_name, tool in self._tools.items():
            desc = f"- {tool_name}: {tool.schema.description}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)


class LLMDecisionEngine:
    """
    LLM决策引擎
    负责与LLM交互并执行工具调用
    """
    
    def __init__(self, openai_client, model_name: str = "gpt-4"):
        self.client = openai_client
        self.model_name = model_name
        self.tool_registry = LLMToolRegistry()
        self.conversation_history = []
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本中的无效Unicode字符"""
        if not isinstance(text, str):
            return text
        try:
            # 移除代理字符
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            return text
    
    @staticmethod
    def _filter_tool_result_for_history(tool_result: Dict) -> Dict:
        """过滤工具结果，移除或截断大段内容以避免历史记录过长
        
        Args:
            tool_result: 原始工具结果
            
        Returns:
            过滤后的工具结果
        """
        filtered_result = tool_result.copy()
        content = filtered_result.get("content", "")
        
        # 如果内容不是字符串，直接返回
        if not isinstance(content, str):
            return filtered_result
        
        # 尝试解析content为JSON（因为工具返回通常是JSON字符串）
        try:
            content_obj = json.loads(content)
            needs_filtering = False
            
            # 检测是否包含需要过滤的大段内容
            if isinstance(content_obj, dict):
                # 提取关键路径字段（这些必须保留）
                critical_fields = {}
                for key in ["script_path", "glb_path", "image_path", "mesh_path", 
                           "output_path", "success", "uid", "output_dir"]:
                    if key in content_obj:
                        critical_fields[key] = content_obj[key]
                
                # 检查是否需要过滤
                for key, value in content_obj.items():
                    if isinstance(value, str):
                        # 检测API文档内容
                        if "API文档" in value or ("bpy." in value and len(value) > 1000):
                            needs_filtering = True
                            content_obj[key] = f"[{key}内容已省略以节省token，长度: {len(value)} 字符]"
                        # 检测Blender脚本内容
                        elif key in ["script_content", "formatted_docs"] and len(value) > 500:
                            needs_filtering = True
                            content_obj[key] = f"[{key}内容已省略以节省token，长度: {len(value)} 字符]"
                        # 检测超长内容
                        elif len(value) > 2000:
                            needs_filtering = True
                            content_obj[key] = value[:400] + f"\n...[中间{len(value)-800}字符已省略]...\n" + value[-400:]
                    elif isinstance(value, list) and len(value) > 10:
                        # 大型列表（如api_list）只保留摘要
                        needs_filtering = True
                        content_obj[key] = f"[列表包含{len(value)}项，已省略详情]"
                
                if needs_filtering:
                    # 确保关键字段存在
                    content_obj.update(critical_fields)
                    # 添加提示信息
                    content_obj["_filter_note"] = "部分大段内容已过滤以节省token，但关键路径字段已保留"
                    filtered_result["content"] = json.dumps(content_obj, ensure_ascii=False)
                else:
                    # 不需要过滤，保持原样
                    filtered_result["content"] = content
            else:
                # 不是字典类型，保持原样
                filtered_result["content"] = content
                
        except json.JSONDecodeError:
            # 如果不是JSON，使用原来的文本过滤逻辑
            # 检测并截断API文档内容
            if "API文档" in content or ("bpy." in content and len(content) > 1000):
                lines = content.split('\n')
                summary_lines = []
                for line in lines:
                    if any(keyword in line for keyword in [
                        "API文档长度", "总计", "准确率", "脚本路径", "完成", 
                        "成功", "失败", "执行", "生成", "步骤"
                    ]):
                        summary_lines.append(line)
                    if len(summary_lines) >= 10:
                        break
                
                if summary_lines:
                    filtered_result["content"] = "\n".join(summary_lines) + "\n[API文档内容已省略以节省token]"
                else:
                    filtered_result["content"] = "[API文档内容已省略以节省token，操作成功完成]"
            
            # 检测并截断Blender脚本内容
            elif "import bpy" in content or ("def " in content and len(content) > 1000):
                filtered_result["content"] = "[Blender脚本内容已省略以节省token，脚本已成功生成]"
            
            # 检测并截断其他超长内容
            elif len(content) > 2000:
                filtered_result["content"] = content[:800] + "\n\n[内容过长，中间部分已省略]\n\n" + content[-400:]
        
        return filtered_result
    
    def register_tool(self, tool: LLMTool):
        """注册工具"""
        self.tool_registry.register(tool)
    
    def register_tools(self, tools: List[LLMTool]):
        """批量注册工具"""
        for tool in tools:
            self.register_tool(tool)
    
    def decide_and_execute_continuous(self, user_input: str, 
                                     max_rounds: int = 5,
                                     interactive: bool = False) -> Dict[str, Any]:
        """
        连续决策和执行，直到LLM不再调用工具或达到最大轮次
        
        Args:
            user_input: 用户输入
            max_rounds: 最大执行轮次
            interactive: 是否启用交互模式（每步后询问用户）
            
        Returns:
            Dict: 包含所有轮次的决策和执行结果
        """
        all_tool_calls = []
        all_responses = []
        task_plan = None  # 存储任务计划
        expected_steps = None  # 期望的步骤数
        completed_step_count = 0  # 已完成的工作步骤数（不包括parse_user_intent和create_task_plan）
        original_user_input = user_input  # 保存原始用户输入，用于重新规划时的上下文
        
        # 如果是交互模式，在初始输入中说明
        if interactive:
            current_input = f"{user_input}\n\n注意：当前为交互模式，每完成一个工作步骤后只说'步骤完成'，不要预告下一步。"
        else:
            current_input = user_input
        
        for round_num in range(max_rounds):
            print(f"\n执行轮次 {round_num + 1}/{max_rounds}")
            print("-" * 50)
            
            result = self.decide_and_execute(current_input, interactive=interactive)
            all_responses.append(result)
            
            # 显示工具调用结果（不重复显示LLM响应，因为在decide_and_execute中已经显示过）
            if result.get("tool_calls"):
                print(f"\n本轮工具调用: {len(result['tool_calls'])}个")
                
                for i, tool_call in enumerate(result["tool_calls"], 1):
                    tool_name = tool_call.get("name", "Unknown")
                    content = tool_call.get("content", "")
                    
                    # 判断工具执行状态
                    success = False
                    try:
                        if content.startswith("{") and content.endswith("}"):
                            tool_result = json.loads(content)
                            success = tool_result.get("success", False)
                            
                        # 提取任务计划（如果是create_task_plan工具）
                        if tool_name == "create_task_plan" and success:
                            task_plan = tool_result.get("execution_plan", {})
                            # 优先从execution_plan中获取，如果没有则从顶层获取
                            expected_steps = task_plan.get("total_steps", None) or tool_result.get("total_steps", None)
                            # 保存计划到缓存
                            try:
                                from tools.intent_parser_tools import set_current_plan
                                set_current_plan(task_plan)
                            except:
                                pass
                            
                            # 立即打印计划信息，确保前端能接收到
                            if expected_steps:
                                print(f"\n   检测到任务计划: 共{expected_steps}个步骤")
                        else:
                            success = "Error" not in content
                    except Exception as e:
                        success = "Error" not in content
                    
                    status_icon = "Success" if success else "Failed"
                    print(f"   {i}. {tool_name}: {status_icon} {'成功' if success else '失败'}")
                    
                    if not success:
                        print(f"      错误: {content[:100]}...")
                
                all_tool_calls.extend(result["tool_calls"])
                
                # 显示最终响应（避免重复）
                final_response = result.get("final_response", "")
                initial_content = result.get("content", "")
                
                # 只在有新内容且与初始内容不同时打印
                if final_response and final_response.strip() and final_response != initial_content:
                    # 检查是否与之前已打印的内容重复
                    if not (final_response in initial_content or initial_content in final_response):
                        print(f"\nLLM继续: {final_response}")
                
                # 检查本轮是否完成了实际工作步骤（不包括parse和plan）
                current_round_work_tools = [
                    tc.get("name") for tc in result["tool_calls"] 
                    if tc.get("name") not in ["parse_user_intent", "create_task_plan"] 
                    and "Error" not in tc.get("content", "")
                ]
                if current_round_work_tools:
                    completed_step_count += len(current_round_work_tools)
                
                # 交互模式：每轮都询问用户（不论是否完成工作步骤）
                user_wants_to_continue = True  # 默认继续
                if interactive and result["tool_calls"]:  # 只要有工具调用就询问
                    print(f"\n{'='*60}")
                    
                    # 显示本轮完成的工具
                    all_current_tools = [tc.get("name") for tc in result["tool_calls"]]
                    if current_round_work_tools:
                        print(f"工作步骤完成: {', '.join(current_round_work_tools)}")
                    else:
                        print(f"准备工作完成: {', '.join(all_current_tools)}")
                    
                    print(f"进度: {completed_step_count}/{expected_steps if expected_steps else '?'} 个工作步骤")
                    print(f"{'='*60}")
                    user_choice = input("\n请选择操作:\n  [回车/y] 继续下一步\n  [n] 停止执行\n  [r] 重新规划\n> ").strip().lower()
                    
                    if user_choice in ['n', 'no', '停止']:
                        print("\n用户选择停止执行")
                        user_wants_to_continue = False
                        break
                    elif user_choice in ['r', 'replan', '重新规划']:
                        print("\n请描述您的需求（LLM会自动识别意图）:")
                        print("  例如: '重做优化提示词' / '重新生成图片' / '不要生成3D模型'")
                        user_input_text = input("> ").strip()
                        
                        if not user_input_text:
                            print("未输入内容，继续执行原计划")
                            continue
                        
                        # 清理用户输入
                        user_input_text = self._clean_text(user_input_text)
                        
                        # 使用LLM分析用户意图
                        print("\n分析用户意图...")
                        completed_tools = [tc.get("name") for tc in all_tool_calls if "Error" not in tc.get("content", "")]
                        
                        # 构建当前任务状态信息
                        task_status = f"""
原始需求: {original_user_input}
当前任务计划: {expected_steps}个步骤
已完成的步骤: {', '.join(completed_tools) if completed_tools else '无'}
"""
                        if task_plan and task_plan.get("steps"):
                            task_status += "\n步骤详情:\n"
                            for i, step in enumerate(task_plan.get("steps", []), 1):
                                step_name = step.get("name", "")
                                step_tool = step.get("tool", "")
                                status = "✓" if step_tool in completed_tools else "○"
                                task_status += f"  {status} {i}. {step_name} ({step_tool})\n"
                        
                        intent_analysis_prompt = f"""
分析用户的重新规划意图。

{task_status}

用户输入: {user_input_text}

请分析用户想要做什么，并以JSON格式返回：
{{
    "intent_type": "redo_step" | "adjust_constraints" | "complete_replan",
    "target_step_tool": "步骤工具名称（仅redo_step时需要）",
    "target_step_number": 步骤编号（仅redo_step时需要）,
    "new_requirement": "调整后的需求描述",
    "reasoning": "判断理由"
}}

intent_type说明：
- redo_step: 用户想重做某个特定步骤（例如"重做优化"、"重新生成图片"）
- adjust_constraints: 用户想调整需求约束（例如"不要生成3D"、"只生成图片"）
- complete_replan: 用户想完全重新规划（提供了全新的需求主题）

只返回JSON，不要其他内容。
"""
                        
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {"role": "system", "content": "你是一个意图分析助手，精确分析用户的重新规划意图。"},
                                    {"role": "user", "content": intent_analysis_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=500
                            )
                            
                            intent_result = response.choices[0].message.content.strip()
                            # 提取JSON（如果有markdown代码块）
                            if "```json" in intent_result:
                                intent_result = intent_result.split("```json")[1].split("```")[0].strip()
                            elif "```" in intent_result:
                                intent_result = intent_result.split("```")[1].split("```")[0].strip()
                            
                            intent_data = json.loads(intent_result)
                            
                            print(f"意图分析结果: {intent_data['intent_type']}")
                            print(f"理由: {intent_data['reasoning']}")
                            
                        except Exception as e:
                            print(f"意图分析失败: {e}，使用默认逻辑")
                            intent_data = {"intent_type": "adjust_constraints", "new_requirement": user_input_text}
                        
                        # 根据意图类型执行不同操作
                        if intent_data["intent_type"] == "redo_step":
                            # 重做某个步骤：回退到该步骤，保留之前的进度
                            target_tool = intent_data.get("target_step_tool")
                            target_number = intent_data.get("target_step_number")
                            
                            if not target_tool and task_plan and task_plan.get("steps"):
                                # 尝试通过步骤编号找到工具
                                steps = task_plan.get("steps", [])
                                if target_number and 1 <= target_number <= len(steps):
                                    target_tool = steps[target_number - 1].get("tool")
                            
                            if target_tool:
                                print(f"\n回退到步骤: {target_tool}")
                                
                                # 找到该步骤在计划中的位置
                                steps = task_plan.get("steps", []) if task_plan else []
                                target_index = -1
                                for i, step in enumerate(steps):
                                    if step.get("tool") == target_tool:
                                        target_index = i
                                        break
                                
                                if target_index >= 0:
                                    # 回退该步骤及之后所有步骤
                                    steps_to_rollback = [step.get("tool") for step in steps[target_index:]]
                                    
                                    print(f"将回退以下步骤: {', '.join(steps_to_rollback)}")
                                    
                                    # 从已完成列表中移除这些步骤
                                    try:
                                        from tools.intent_parser_tools import unmark_step_completed
                                        for step_tool in steps_to_rollback:
                                            unmark_step_completed(step_tool)
                                    except:
                                        pass
                                    
                                    # 从工具调用历史中移除这些步骤
                                    all_tool_calls = [
                                        tc for tc in all_tool_calls 
                                        if tc.get("name") not in steps_to_rollback
                                    ]
                                    
                                    # 重新计算已完成步骤数
                                    completed_step_count = len([
                                        tc for tc in all_tool_calls 
                                        if tc.get("name") not in ["parse_user_intent", "create_task_plan"]
                                        and "Error" not in tc.get("content", "")
                                    ])
                                    
                                    print(f"已回退完成，当前进度: {completed_step_count}/{expected_steps}")
                                    
                                    # 构建继续执行的输入
                                    mode_instruction = ""
                                    if interactive:
                                        mode_instruction = "\n注意：当前为交互模式，完成工具调用后只说'步骤完成'，不要预告下一步。"
                                    
                                    completed_tools = [tc.get("name") for tc in all_tool_calls if "Error" not in tc.get("content", "")]
                                    next_step = steps[target_index]
                                    
                                    current_input = f"""
继续执行任务计划（已回退到指定步骤）。

原始需求: {original_user_input}
任务计划: {expected_steps}个步骤
已完成: {', '.join(completed_tools)}
下一步: {next_step.get('name')} ({next_step.get('tool')})

请从此步骤重新开始执行。{mode_instruction}

立即执行 {next_step.get('tool')} 工具。
"""
                                    print(f"准备重新执行: {next_step.get('name')}")
                                    continue
                                else:
                                    print(f"错误: 未找到步骤 {target_tool}")
                            else:
                                print("错误: 无法识别要重做的步骤")
                        
                        elif intent_data["intent_type"] == "complete_replan":
                            # 完全重新规划
                            new_requirement = intent_data.get("new_requirement", user_input_text)
                            
                            print(f"\n完全重新规划任务")
                            print(f"新需求: {new_requirement}")
                            
                            # 更新原始输入
                            original_user_input = new_requirement
                            
                            # 【关键修改】在清空历史之前，先从all_tool_calls中提取并保存关键路径到会话memory
                            try:
                                from tools.intent_parser_tools import get_session_memory
                                key_results = self._extract_key_results_from_tool_calls(all_tool_calls)
                                session_memory = get_session_memory()
                                print(f"保留会话memory: uid={session_memory.get('uid')}, output_dir={session_memory.get('output_dir')}")
                            except Exception as e:
                                print(f"提取会话memory失败: {e}")
                            
                            # 清空意图缓存和步骤状态，重新解析
                            try:
                                from tools.intent_parser_tools import clear_intent_cache, clear_step_status
                                clear_intent_cache()
                                clear_step_status()
                                # 注意：不清空 clear_session_memory()，会话信息永久保留！
                            except:
                                pass
                            
                            # 重置所有状态并重新开始
                            task_plan = None
                            expected_steps = None
                            completed_step_count = 0
                            all_tool_calls = []  # 清空工具调用历史
                            all_responses = []   # 清空响应历史
                            self.clear_history()  # 清空LLM对话历史
                            
                            mode_instruction = ""
                            if interactive:
                                mode_instruction = "\n\n注意：当前为交互模式，每完成一个工作步骤后只说'步骤完成'，不要预告下一步。"
                            
                            current_input = f"""
用户要求完全重新规划任务。

全新需求: {new_requirement}

请重新执行:
1. 调用 parse_user_intent 解析新的需求
2. 调用 create_task_plan 制定新的计划
3. 开始执行新计划{mode_instruction}
"""
                            print("清空历史记录，从头开始...")
                            continue
                        
                        else:  # adjust_constraints
                            # 调整需求约束
                            adjustment = intent_data.get("new_requirement", user_input_text)
                            
                            print(f"\n调整需求约束")
                            print(f"原始需求: {original_user_input}")
                            print(f"用户调整: {adjustment}")
                            
                            # 【关键修改】在清空历史之前，先从all_tool_calls中提取并保存关键路径到会话memory
                            try:
                                from tools.intent_parser_tools import get_session_memory
                                key_results = self._extract_key_results_from_tool_calls(all_tool_calls)
                                session_memory = get_session_memory()
                                print(f"保留会话memory: uid={session_memory.get('uid')}, output_dir={session_memory.get('output_dir')}")
                            except Exception as e:
                                print(f"提取会话memory失败: {e}")
                            
                            # 清空意图缓存和步骤状态，重新解析
                            try:
                                from tools.intent_parser_tools import clear_intent_cache, clear_step_status
                                clear_intent_cache()
                                clear_step_status()
                                # 注意：不清空 clear_session_memory()，会话信息永久保留！
                            except:
                                pass
                            
                            # 重置所有状态并重新开始
                            task_plan = None
                            expected_steps = None
                            completed_step_count = 0
                            all_tool_calls = []  # 清空工具调用历史
                            all_responses = []   # 清空响应历史
                            self.clear_history()  # 清空LLM对话历史
                            
                            mode_instruction = ""
                            if interactive:
                                mode_instruction = "\n\n注意：当前为交互模式，每完成一个工作步骤后只说'步骤完成'，不要预告下一步。"
                            
                            current_input = f"""
用户要求调整任务约束。

原始需求: {original_user_input}
用户调整: {adjustment}

请基于原始需求和用户调整，重新执行:
1. 调用 parse_user_intent 解析调整后的需求（保留原始需求的主题，应用新的约束条件）
2. 调用 create_task_plan 制定新的计划
3. 开始执行新计划{mode_instruction}
"""
                            print("重新规划任务...")
                            continue
                    # else: 用户选择继续（回车或y）
                    # 标记刚完成的步骤（包括所有工具，不只是工作步骤）
                    try:
                        from tools.intent_parser_tools import mark_step_completed
                        for tc in result["tool_calls"]:
                            step_name = tc.get("name")
                            if "Error" not in tc.get("content", ""):
                                mark_step_completed(step_name)
                    except:
                        pass
                    print("\n继续执行下一步...")
                
                # 判断是否需要继续执行
                total_tools_called = len(all_tool_calls)
                completed_tools = [tc.get("name") for tc in all_tool_calls if "Error" not in tc.get("content", "")]
                
                # 如果还没有创建任务计划，必须继续
                if "create_task_plan" not in completed_tools:
                    should_continue = True
                    print(f"   还未创建任务计划，需要继续执行")
                elif expected_steps is not None:
                    # 如果有任务计划，根据计划的步骤数判断
                    if completed_step_count < expected_steps:
                        should_continue = True
                        print(f"   已完成 {completed_step_count}/{expected_steps} 个任务步骤")
                    else:
                        print(f"   已完成所有 {expected_steps} 个任务步骤")
                        should_continue = False
                else:
                    # 如果没有任务计划，使用默认逻辑（5个工具）
                    if total_tools_called < 5:
                        should_continue = True
                        print(f"   已调用 {total_tools_called}/5 个工具")
                    else:
                        should_continue = False
                
                # 检查是否有重复的工具调用（防止无限循环）
                recent_tools = [tc.get("name") for tc in result["tool_calls"]]
                if len(set(recent_tools)) < len(recent_tools):
                    print(f"检测到重复工具调用，停止执行")
                    break
                
                if should_continue and round_num < max_rounds - 1 and user_wants_to_continue:
                    # 分析当前执行状态，构建更准确的下一轮输入
                    completed_tools = [tc.get("name") for tc in all_tool_calls if "Error" not in tc.get("content", "")]
                    
                    print(f"   准备构建下一轮输入...")
                    print(f"   当前状态: task_plan={'有' if task_plan else '无'}, completed_tools={completed_tools}")
                    
                    # 如果有任务计划，使用计划中的步骤
                    if task_plan and task_plan.get("steps"):
                        steps = task_plan.get("steps", [])
                        next_step = None
                        
                        # 找到下一个未完成的步骤
                        for step in steps:
                            step_tool = step.get("tool")
                            if step_tool not in completed_tools:
                                next_step = step
                                break
                        
                        if next_step:
                            next_tool = next_step.get("tool")
                            step_name = next_step.get("name")
                            
                            mode_instruction = ""
                            if interactive:
                                mode_instruction = "\n注意：当前为交互模式，完成工具调用后只说'步骤完成'，不要预告下一步。"
                            
                            current_input = f"""
根据任务计划，继续执行下一步。

任务计划: {expected_steps}个步骤
已完成: {', '.join(completed_tools)}
下一步: {step_name} ({next_tool})

CRITICAL: 严格遵循任务计划，不要执行计划外的步骤！
任务计划只有{expected_steps}个步骤，完成后即停止。{mode_instruction}

立即执行 {next_tool} 工具。
"""
                        else:
                            # 所有计划步骤都完成了
                            print(f"所有计划步骤已完成")
                            break
                    else:
                        # 没有任务计划，确定下一个要调用的工具
                        mode_instruction = ""
                        if interactive:
                            mode_instruction = "\n注意：当前为交互模式，完成工具调用后只说'步骤完成'，不要预告下一步。"
                        
                        # 如果还没有parse_user_intent，先调用它
                        if "parse_user_intent" not in completed_tools:
                            current_input = f"""
第一步：必须先解析用户意图。

立即调用 parse_user_intent 工具来理解用户的真实需求。{mode_instruction}
"""
                        elif "create_task_plan" not in completed_tools:
                            current_input = f"""
第二步：基于解析的意图创建任务计划。

调用 create_task_plan 工具（工具会自动使用最新的解析意图）。{mode_instruction}
"""
                        else:
                            # 使用默认工具列表
                            all_required_tools = ["parse_user_intent", "create_task_plan", "optimize_3d_prompt", "text_to_image", "img_to_3d_complete", "render_3d_scene"]
                            next_tool = None
                            for tool in all_required_tools:
                                if tool not in completed_tools:
                                    next_tool = tool
                                    break
                            
                            current_input = f"""
继续3D生成任务。

已完成: {', '.join(completed_tools)}
下一个必须执行: {next_tool}

要求：
1. 立即调用 {next_tool} 工具
2. 不要只是描述，必须实际调用工具
3. 调用后继续执行后续工具{mode_instruction}

立即执行 {next_tool} 工具。
"""
                    
                    print(f"检测到需要继续执行，进入下一轮...")
                    print(f"   已完成工具: {completed_tools}")
                else:
                    print(f"执行完成或无需继续")
                    break
            else:
                print(f"本轮无工具调用")
                if result.get("content"):
                    print(f"LLM响应: {result['content']}")
                
                # 如果LLM没有调用工具，但还有步骤未完成，给出明确指示
                if task_plan and expected_steps and completed_step_count < expected_steps:
                    print(f"\n警告: LLM未调用工具，但任务未完成 ({completed_step_count}/{expected_steps})")
                    
                    # 找到下一个未完成的步骤
                    completed_tools = [tc.get("name") for tc in all_tool_calls if "Error" not in tc.get("content", "")]
                    steps = task_plan.get("steps", [])
                    next_step = None
                    
                    for step in steps:
                        step_tool = step.get("tool")
                        if step_tool not in completed_tools:
                            next_step = step
                            break
                    
                    if next_step and round_num < max_rounds - 1:
                        next_tool = next_step.get("tool")
                        step_name = next_step.get("name")
                        
                        # 提取前面步骤的关键返回值
                        key_results = self._extract_key_results_from_tool_calls(all_tool_calls)
                        results_info = ""
                        if key_results:
                            results_info = "\n\n前面步骤的关键返回值（请使用这些实际值作为参数）:\n"
                            for key, value in key_results.items():
                                results_info += f"- {key}: {value}\n"
                        
                        mode_instruction = ""
                        if interactive:
                            mode_instruction = "\n注意：当前为交互模式，完成工具调用后只说'步骤完成'，不要预告下一步。"
                        
                        print(f"自动进入下一轮，执行下一步: {step_name}")
                        current_input = f"""
你刚才没有调用任何工具。请立即调用工具来执行任务。

任务计划: {expected_steps}个步骤
已完成: {', '.join(completed_tools)}
下一步必须执行: {step_name} ({next_tool}){results_info}

CRITICAL 重要：
1. 不要只在文本中描述要做什么
2. 必须实际调用 {next_tool} 工具（使用tool_calls机制）
3. 不要使用 <tool_call> 这样的文本标签
4. 使用上面提供的实际返回值作为参数，不要猜测或编造{mode_instruction}

立即调用 {next_tool} 工具。
"""
                        continue
                    else:
                        break
                else:
                    break
        
        return {
            "total_rounds": round_num + 1,
            "all_tool_calls": all_tool_calls,
            "all_responses": all_responses,
            "final_response": all_responses[-1].get("final_response", "") if all_responses else "",
            "success": len(all_tool_calls) > 0
        }

    def decide_and_execute(self, user_input: str, 
                          system_prompt: Optional[str] = None,
                          interactive: bool = False) -> Dict[str, Any]:
        """
        让LLM决策并执行工具调用
        
        Args:
            user_input: 用户输入
            system_prompt: 系统提示
            interactive: 是否在交互模式（会限制每次只执行一个工作步骤）
            
        Returns:
            Dict: 包含决策和执行结果
        """
        # 构建系统提示
        if system_prompt is None:
            system_prompt = self._build_system_prompt()
        
        # 构建消息
        messages = [
            {"role": "system", "content": self._clean_text(system_prompt)},
            {"role": "user", "content": self._clean_text(user_input)}
        ]
        
        # 添加历史对话（清理所有内容，确保 content 不为 None）
        if self.conversation_history:
            cleaned_history = []
            for msg in self.conversation_history:
                cleaned_msg = msg.copy()
                if "content" in cleaned_msg:
                    if cleaned_msg["content"] is None:
                        # 如果 content 是 None，设置为空字符串
                        cleaned_msg["content"] = ""
                    elif cleaned_msg["content"]:
                        cleaned_msg["content"] = self._clean_text(cleaned_msg["content"])
                cleaned_history.append(cleaned_msg)
            messages = [messages[0]] + cleaned_history + [messages[-1]]
        
        # 调用LLM - 强制使用工具
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_registry.get_function_definitions(),
                tool_choice="auto",
                max_tokens=4000,  # 增加token限制以支持更长的响应
                temperature=0.1   # 降低随机性，确保一致的执行模式
            )
        except UnicodeEncodeError as e:
            print(f"   Unicode编码错误，清空历史后重试: {e}")
            # 如果仍然有编码错误，完全不使用历史
            self.clear_history()
            messages = [
                {"role": "system", "content": self._clean_text(system_prompt)},
                {"role": "user", "content": self._clean_text(user_input)}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_registry.get_function_definitions(),
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.1
            )
        
        # 处理响应
        message = response.choices[0].message
        
        # 记录到历史（清理文本）
        self.conversation_history.append({
            "role": "user",
            "content": self._clean_text(user_input)
        })
        
        result = {
            "content": message.content,
            "tool_calls": []
        }
        
        # 显示LLM的初始响应
        if message.content and message.content.strip():
            print(f"LLM: {message.content}")
        
        # 如果有工具调用
        if message.tool_calls:
            # 在交互模式或初始化阶段，每轮只执行一个工具
            if interactive or len(message.tool_calls) > 1:
                # 检查是否包含parse_user_intent或create_task_plan
                first_tool_name = message.tool_calls[0].function.name if hasattr(message.tool_calls[0], 'function') else None
                
                if len(message.tool_calls) > 1:
                    print(f"检测到多个工具调用（{len(message.tool_calls)}个），只执行第一个: {first_tool_name}")
                    # 只保留第一个工具调用
                    tool_calls_to_process = [message.tool_calls[0]]
                else:
                    tool_calls_to_process = message.tool_calls
            else:
                tool_calls_to_process = message.tool_calls
            
            tool_results = self.tool_registry.process_tool_calls(tool_calls_to_process)
            result["tool_calls"] = tool_results
            
            # 将工具调用转换为可序列化的格式用于历史记录（只记录实际处理的）
            tool_calls_dict = []
            for tc in tool_calls_to_process:
                if hasattr(tc, 'function'):
                    # OpenAI SDK对象
                    tool_calls_dict.append({
                        "id": getattr(tc, 'id', None),
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                elif isinstance(tc, dict):
                    tool_calls_dict.append(tc)
            
            # 将工具调用结果添加到历史（清理文本）
            # 构建历史记录消息，确保 content 不为 None
            history_msg = {
                "role": "assistant",
                "tool_calls": tool_calls_dict
            }
            # 只有当 content 不为 None 时才添加 content 字段（避免 API 报错）
            if message.content is not None:
                history_msg["content"] = self._clean_text(message.content)
            else:
                history_msg["content"] = ""  # 使用空字符串而不是 None
            
            self.conversation_history.append(history_msg)
            
            # 过滤工具结果以避免历史记录过长
            filtered_tool_results = []
            for tool_result in tool_results:
                filtered_result = self._filter_tool_result_for_history(tool_result)
                self.conversation_history.append(filtered_result)
                filtered_tool_results.append(filtered_result)
            
            # 获取最终响应
            # 构建 assistant 消息，确保 content 不为 None（如果只有工具调用，使用空字符串）
            assistant_message = {
                "role": "assistant",
                "tool_calls": tool_calls_dict
            }
            # 只有当 content 不为 None 时才添加 content 字段
            if message.content is not None:
                assistant_message["content"] = message.content
            else:
                # 如果 content 为 None，设置为空字符串（某些 API 要求）
                assistant_message["content"] = ""
            
            # 使用过滤后的工具结果来构建消息，避免token超限
            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages + [assistant_message] + filtered_tool_results,
                tools=self.tool_registry.get_function_definitions(),
                tool_choice="auto"
            )
            
            final_message = final_response.choices[0].message
            result["final_response"] = final_message.content
            
            # 打印LLM在工具调用后的响应
            if final_message.content and final_message.content.strip():
                print(f"\nLLM: {final_message.content}")
            
            # 检查是否有进一步的工具调用
            if final_message.tool_calls:
                # 在交互模式下，每轮只执行一个工具，不再继续
                if interactive:
                    print("交互模式：本轮已执行工具，等待用户确认后进入下一轮")
                    additional_results = []
                else:
                    # 非交互模式，正常处理后续工具调用
                    print("LLM请求继续执行工具...")
                    
                    # 检查是否是重复调用
                    new_tool_names = [tc.function.name for tc in final_message.tool_calls]
                    existing_tool_names = [tc.get("name") for tc in tool_results]
                    
                    if any(name in existing_tool_names for name in new_tool_names):
                        print("检测到重复工具调用，跳过以避免循环")
                        additional_results = []  # 初始化为空列表
                    else:
                        # 递归处理后续工具调用
                        additional_results = self.tool_registry.process_tool_calls(final_message.tool_calls)
                        result["tool_calls"].extend(additional_results)
                
                # 只有当有新的工具调用结果时才获取最终响应
                if additional_results:
                    # 构建第一个 assistant 消息（确保 content 不为 None）
                    first_assistant_msg = {
                        "role": "assistant",
                        "tool_calls": tool_calls_dict
                    }
                    if message.content is not None:
                        first_assistant_msg["content"] = message.content
                    else:
                        first_assistant_msg["content"] = ""
                    
                    # 构建第二个 assistant 消息（确保 content 不为 None）
                    second_assistant_msg = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": getattr(tc, 'id', None),
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in final_message.tool_calls
                        ]
                    }
                    if final_message.content is not None:
                        second_assistant_msg["content"] = final_message.content
                    else:
                        second_assistant_msg["content"] = ""
                    
                    # 过滤additional_results
                    filtered_additional_results = [self._filter_tool_result_for_history(r) for r in additional_results]
                    
                    # 再次获取最终响应，使用过滤后的结果避免token超限
                    final_final_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages + [first_assistant_msg] + filtered_tool_results + [second_assistant_msg] + filtered_additional_results
                    )
                else:
                    # 如果没有新的工具调用，直接使用当前的final_message
                    final_final_response = final_response
                
                result["final_response"] = final_final_response.choices[0].message.content
                
                # 打印最终的最终响应（如果有额外工具调用）
                if additional_results and final_final_response.choices[0].message.content:
                    final_content = final_final_response.choices[0].message.content.strip()
                    if final_content:
                        print(f"\nLLM: {final_content}")
                
                # 记录所有对话（确保 content 不为 None）
                if additional_results:
                    # 构建第一个 assistant 历史消息
                    first_history_msg = {
                        "role": "assistant",
                        "content": self._clean_text(final_message.content) if final_message.content else "",
                        "tool_calls": [
                            {
                                "id": getattr(tc, 'id', None),
                                "type": "function", 
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in final_message.tool_calls
                        ]
                    }
                    # 构建第二个 assistant 历史消息
                    second_history_msg = {
                        "role": "assistant",
                        "content": self._clean_text(final_final_response.choices[0].message.content) if final_final_response.choices[0].message.content else ""
                    }
                    # 使用过滤后的additional_results添加到历史记录
                    self.conversation_history.extend([first_history_msg] + filtered_additional_results + [second_history_msg])
                else:
                    # 没有额外的工具调用，只记录当前响应
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": self._clean_text(final_message.content) if final_message.content else ""
                    })
            else:
                # 记录最终响应
                self.conversation_history.append({
                    "role": "assistant",
                    "content": self._clean_text(final_message.content) if final_message.content else ""
                })
        else:
            # 记录助手响应
            self.conversation_history.append({
                "role": "assistant",
                "content": self._clean_text(message.content) if message.content else ""
            })
        
        return result
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_desc = self.tool_registry.get_tools_description()
        
        # 从文件加载系统提示模板
        prompt_file = os.path.join(os.path.dirname(__file__), "system_prompt", "llm_decision_engine_prompt")
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            return prompt_template.format(tools_desc=tools_desc)
        except Exception as e:
            print(f"   警告: 无法加载系统提示文件 {prompt_file}: {e}")
            # 后备方案：返回简单的系统提示
            return f"""You are an AI assistant that helps with 3D generation tasks.

You have access to the following tools:
{tools_desc}

Please follow the task plan and execute tools one by one.
"""
    
    def _extract_key_results_from_tool_calls(self, tool_calls: List[Dict]) -> Dict[str, str]:
        """
        从工具调用结果中提取关键返回值
        
        Args:
            tool_calls: 工具调用结果列表
            
        Returns:
            Dict: 关键返回值字典
        """
        key_results = {}
        
        for tool_call in tool_calls:
            content = tool_call.get("content", "")
            
            # 尝试解析JSON内容
            try:
                if isinstance(content, str) and content.startswith("{"):
                    result = json.loads(content)
                    
                    # 提取关键路径字段
                    for key in ["image_path", "glb_path", "script_path", "mesh_path", 
                               "output_path", "uid", "output_dir", "session_dir"]:
                        if key in result and result[key]:
                            key_results[key] = result[key]
            except:
                pass
        
        return key_results
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        # 确保是一个新的空列表
        import gc
        gc.collect()
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_history.copy()


# 全局LLM工具注册实例
llm_tool_registry = LLMToolRegistry()
