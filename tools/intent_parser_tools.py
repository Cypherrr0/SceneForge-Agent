"""
用户意图解析工具
使用LLM智能解析用户请求，提取任务约束和要求
"""
import json
import os
from typing import Dict, Any, List
from openai import OpenAI

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


# 全局意图缓存 - 用于在工具间传递解析结果
_INTENT_CACHE = {
    "latest_intent": None
}

# 全局步骤完成状态缓存
_STEP_STATUS_CACHE = {
    "completed_steps": [],
    "current_plan": None
}

# 全局会话级别持久化memory - 存储关键路径信息，重做时不清除
_SESSION_MEMORY = {
    "uid": None,
    "output_dir": None,
    "session_dir": None,
    "base_save_dir": None,
    "created_at": None
}


class ParseUserIntentTool(LLMTool):
    """解析用户意图工具 - 使用LLM智能理解用户需求"""
    
    def __init__(self, openai_client: OpenAI = None, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.client = openai_client
        self.model_name = model_name
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="parse_user_intent",
            description="Parse user request to extract intent, constraints, and requirements using LLM",
            parameters=[
                ToolParameter(
                    name="user_request",
                    type="string",
                    description="Original user request in natural language"
                )
            ],
            returns="Dict with parsed intent including what user wants and what to skip",
            category="planning"
        )
    
    def execute(self, user_request: str) -> Dict[str, Any]:
        """使用LLM解析用户意图"""
        
        if not self.client:
            # 如果没有LLM客户端，返回默认解析（向后兼容）
            return self._fallback_parse(user_request)
        
        # 从文件加载解析提示词
        prompt_file = os.path.join(os.path.dirname(__file__), "system_prompt", "intent_parser_prompt")
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            parse_prompt = prompt_template.format(user_request=user_request)
        except Exception as e:
            print(f"   无法加载提示词文件 {prompt_file}: {e}")
            return self._fallback_parse(user_request)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise task intent parser. Always respond with valid JSON only."},
                    {"role": "user", "content": parse_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 提取JSON（如果LLM返回了额外的文字）
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(result_text)
            
            result = {
                "success": True,
                "intent": parsed,
                "generation_prompt": parsed.get("generation_prompt", user_request),
                "wants": parsed.get("wants", {}),
                "constraints": parsed.get("constraints", {}),
                "rendering_complexity": parsed.get("rendering_complexity", {
                    "needs_coder_tools": False,
                    "complexity_type": "simple",
                    "description": "标准渲染"
                }),
                "reason": parsed.get("reason", "")
            }
            
            # 存储到全局缓存，供create_task_plan使用
            _INTENT_CACHE["latest_intent"] = result
            
            return result
            
        except Exception as e:
            print(f"   LLM解析失败: {e}, 使用后备解析")
            return self._fallback_parse(user_request)
    
    def _fallback_parse(self, user_request: str) -> Dict[str, Any]:
        """后备解析方法（基于规则）"""
        import re
        
        # 简单的字符串匹配作为后备
        request_lower = user_request.lower()
        
        wants = {
            "optimize_prompt": True,
            "generate_2d": True,
            "generate_3d": True,
            "render_views": True
        }
        
        constraints = {
            "only_2d": False,
            "skip_optimize": False,
            "skip_2d": False,
            "skip_3d": False,
            "skip_render": False
        }
        
        # 检测各种约束
        if any(phrase in user_request for phrase in ["只生成2D", "仅生成2D", "只要2D", "只生成图像"]):
            wants["generate_3d"] = False
            wants["render_views"] = False
            constraints["only_2d"] = True
        
        if any(phrase in user_request for phrase in ["不要优化", "不需要优化", "跳过优化"]):
            wants["optimize_prompt"] = False
            constraints["skip_optimize"] = True
        
        if any(phrase in user_request for phrase in ["不要生成2D", "不生成2D", "不要2D图像", "不生成图像"]):
            wants["generate_2d"] = False
            constraints["skip_2d"] = True
        
        if any(phrase in user_request for phrase in ["不要生成3D", "不生成3D", "不要3D模型"]):
            wants["generate_3d"] = False
            wants["render_views"] = False
            constraints["skip_3d"] = True
        
        if any(phrase in user_request for phrase in ["不要渲染", "不需要渲染", "跳过渲染"]):
            wants["render_views"] = False
            constraints["skip_render"] = True
        
        # 检测复杂渲染需求
        rendering_complexity = {
            "needs_coder_tools": False,
            "complexity_type": "simple",
            "description": "标准渲染"
        }
        
        # 检测摄像机运动相关关键词
        camera_movement_keywords = [
            "推进", "环绕", "拉远", "运镜", "动画", "摄像机", "相机", "镜头",
            "推进", "环绕", "拉远", "运镜", "动画", "摄像机", "相机", "镜头",
            "camera", "movement", "orbit", "zoom", "pan", "dolly", "track"
        ]
        
        # 检测时间相关关键词
        time_keywords = [
            "秒", "分钟", "时间", "停留", "循环", "持续", "秒内", "分钟内",
            "second", "minute", "duration", "stay", "loop", "cycle"
        ]
        
        # 检测复杂场景操作
        complex_scene_keywords = [
            "多个", "组合", "摆放", "排列", "场景", "环境", "背景",
            "multiple", "arrange", "position", "scene", "environment", "background"
        ]
        
        # 检测动画效果
        animation_keywords = [
            "动画", "运动", "旋转", "移动", "变换", "效果",
            "animation", "motion", "rotation", "movement", "transform", "effect"
        ]
        
        request_lower = user_request.lower()
        
        if any(keyword in request_lower for keyword in camera_movement_keywords):
            rendering_complexity["needs_coder_tools"] = True
            rendering_complexity["complexity_type"] = "camera_movement"
            rendering_complexity["description"] = "摄像机运动"
        elif any(keyword in request_lower for keyword in time_keywords):
            rendering_complexity["needs_coder_tools"] = True
            rendering_complexity["complexity_type"] = "animation"
            rendering_complexity["description"] = "时间控制动画"
        elif any(keyword in request_lower for keyword in complex_scene_keywords):
            rendering_complexity["needs_coder_tools"] = True
            rendering_complexity["complexity_type"] = "complex_scene"
            rendering_complexity["description"] = "复杂场景"
        elif any(keyword in request_lower for keyword in animation_keywords):
            rendering_complexity["needs_coder_tools"] = True
            rendering_complexity["complexity_type"] = "animation"
            rendering_complexity["description"] = "动画效果"
        
        # 提取生成提示词
        generation_prompt = self._extract_prompt(user_request)
        
        result = {
            "success": True,
            "intent": {
                "generation_prompt": generation_prompt,
                "wants": wants,
                "constraints": constraints,
                "rendering_complexity": rendering_complexity,
                "reason": "使用规则解析"
            },
            "generation_prompt": generation_prompt,
            "wants": wants,
            "constraints": constraints,
            "rendering_complexity": rendering_complexity,
            "reason": "使用规则解析（后备方案）"
        }
        
        # 存储到全局缓存
        _INTENT_CACHE["latest_intent"] = result
        
        return result
    
    def _extract_prompt(self, request: str) -> str:
        """提取纯粹的生成提示词"""
        import re
        
        # 移除任务约束和技术操作部分
        constraints_patterns = [
            r"[，。,\.]\s*不要.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*只生成.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*不需要.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*跳过.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*首先.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*随后.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*接着.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*最后.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*将.*?置于.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*缩放.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*创建.*?摄像机.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*摄像机.*?运动.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*推进.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*环绕.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*拉远.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*运镜.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*动画.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*秒内.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*停留.*?(?=[，。,\.\n]|$)",
            r"[，。,\.]\s*循环.*?(?=[，。,\.\n]|$)"
        ]
        
        clean_request = request
        for pattern in constraints_patterns:
            clean_request = re.sub(pattern, "", clean_request)
        
        # 查找描述内容 - 优先查找视觉描述
        # 尝试找到第一个句号前的内容（通常是视觉描述）
        sentences = clean_request.split('。')
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # 如果第一句包含视觉描述词汇，使用它
            visual_keywords = ['人', '物', '动物', '椅子', '桌子', '花瓶', '猫', '狗', '大象', '马戏', '表演', '舞台', '观看']
            if any(keyword in first_sentence for keyword in visual_keywords):
                return first_sentence
        
        # 查找描述内容
        pattern = r"(?:生成|创建|制作)(?:3D模型)?[：:]?\s*(.+?)(?:\n|$)"
        match = re.search(pattern, clean_request)
        if match:
            return match.group(1).strip()
        
        return clean_request.strip()


def get_latest_intent() -> Dict[str, Any]:
    """获取最新解析的用户意图"""
    return _INTENT_CACHE.get("latest_intent")


def clear_intent_cache():
    """清空意图缓存"""
    _INTENT_CACHE["latest_intent"] = None


def mark_step_completed(step_name: str):
    """标记步骤为已完成"""
    if step_name not in _STEP_STATUS_CACHE["completed_steps"]:
        _STEP_STATUS_CACHE["completed_steps"].append(step_name)


def unmark_step_completed(step_name: str):
    """取消标记步骤为已完成（用于重做步骤）"""
    if step_name in _STEP_STATUS_CACHE["completed_steps"]:
        _STEP_STATUS_CACHE["completed_steps"].remove(step_name)


def get_completed_steps() -> List[str]:
    """获取已完成的步骤列表"""
    return _STEP_STATUS_CACHE["completed_steps"].copy()


def clear_step_status():
    """清空步骤状态"""
    _STEP_STATUS_CACHE["completed_steps"] = []
    _STEP_STATUS_CACHE["current_plan"] = None


def set_current_plan(plan: Dict[str, Any]):
    """设置当前执行计划"""
    _STEP_STATUS_CACHE["current_plan"] = plan


def get_current_plan() -> Dict[str, Any]:
    """获取当前执行计划"""
    return _STEP_STATUS_CACHE.get("current_plan")


# === 会话级别持久化memory管理函数 ===

def set_session_memory(uid: str = None, output_dir: str = None, session_dir: str = None, 
                      base_save_dir: str = None):
    """设置会话级别的持久化memory（重做时不清除）"""
    import datetime
    
    if uid is not None:
        _SESSION_MEMORY["uid"] = uid
    if output_dir is not None:
        _SESSION_MEMORY["output_dir"] = output_dir
    if session_dir is not None:
        _SESSION_MEMORY["session_dir"] = session_dir
    if base_save_dir is not None:
        _SESSION_MEMORY["base_save_dir"] = base_save_dir
    
    # 记录创建时间
    if _SESSION_MEMORY["created_at"] is None:
        _SESSION_MEMORY["created_at"] = datetime.datetime.now().isoformat()


def get_session_memory() -> Dict[str, Any]:
    """获取会话级别的持久化memory"""
    return _SESSION_MEMORY.copy()


def update_session_memory_from_tool_result(tool_result: Dict[str, Any]):
    """从工具执行结果中自动更新会话memory"""
    # 提取关键路径字段
    if "uid" in tool_result and tool_result["uid"]:
        _SESSION_MEMORY["uid"] = tool_result["uid"]
    if "output_dir" in tool_result and tool_result["output_dir"]:
        _SESSION_MEMORY["output_dir"] = tool_result["output_dir"]
        _SESSION_MEMORY["session_dir"] = tool_result["output_dir"]  # session_dir 通常等于 output_dir
    if "save_dir" in tool_result and tool_result["save_dir"]:
        _SESSION_MEMORY["base_save_dir"] = tool_result["save_dir"]


def clear_session_memory():
    """清空会话memory（仅在真正需要新建会话时调用）"""
    _SESSION_MEMORY["uid"] = None
    _SESSION_MEMORY["output_dir"] = None
    _SESSION_MEMORY["session_dir"] = None
    _SESSION_MEMORY["base_save_dir"] = None
    _SESSION_MEMORY["created_at"] = None


def register_intent_parser_tools(registry, openai_client=None, model_name="Qwen/Qwen2.5-72B-Instruct"):
    """
    注册用户意图解析工具
    
    Args:
        registry: LLM工具注册中心
        openai_client: OpenAI客户端
        model_name: LLM模型名称
    """
    tools = [
        ParseUserIntentTool(openai_client, model_name)
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} intent parser tools\n")
    return tools
