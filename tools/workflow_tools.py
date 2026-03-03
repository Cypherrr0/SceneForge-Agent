"""
工作流管理工具
帮助LLM更好地管理和跟踪任务执行步骤
"""
import json
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


@dataclass
class WorkflowStep:
    """工作流步骤定义"""
    id: str
    name: str
    tool_name: str
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict] = None
    error: Optional[str] = None


class CreateWorkflowTool(LLMTool):
    """创建工作流工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="create_workflow",
            description="Create a structured workflow for 3D generation tasks",
            parameters=[
                ToolParameter(
                    name="task_description",
                    type="string",
                    description="Description of the 3D generation task"
                ),
                ToolParameter(
                    name="include_evaluation",
                    type="boolean",
                    description="Whether to include evaluation steps",
                    required=False,
                    default=False
                )
            ],
            returns="Dict with workflow_id and steps",
            category="workflow"
        )
    
    def execute(self, task_description: str, include_evaluation: bool = False) -> Dict[str, Any]:
        """创建3D生成工作流"""
        workflow_id = str(uuid.uuid4())[:8]
        
        # 定义标准3D生成步骤
        steps = [
            WorkflowStep(
                id="step_1",
                name="优化提示词",
                tool_name="optimize_3d_prompt",
                parameters={
                    "original_prompt": task_description,
                    "target_model": "gemini",
                    "translate_to_english": False
                }
            ),
            WorkflowStep(
                id="step_2", 
                name="生成2D参考图像",
                tool_name="text_to_image",
                parameters={
                    "prompt": "{optimized_prompt}",  # 占位符，将从上一步获取
                    "style": "3d"
                }
            ),
            WorkflowStep(
                id="step_3",
                name="生成3D模型",
                tool_name="img_to_3d_complete",
                parameters={
                    "prompt": "{optimized_prompt}",
                    "use_existing_image": "{image_path}"
                }
            ),
            WorkflowStep(
                id="step_4",
                name="渲染多视角图像",
                tool_name="render_3d_scene",
                parameters={
                    "glb_file_path": "{glb_path}",
                    "output_dir": "{output_dir}/renders"
                }
            )
        ]
        
        # 如果需要评估，添加评估步骤
        if include_evaluation:
            steps.extend([
                WorkflowStep(
                    id="step_5",
                    name="生成评估指标",
                    tool_name="generate_evaluation_index",
                    parameters={
                        "prompt": task_description,
                        "reference_image": "{image_path}"
                    }
                ),
                WorkflowStep(
                    id="step_6",
                    name="评估3D模型质量",
                    tool_name="evaluate_3d_asset",
                    parameters={
                        "evaluation_index": "{evaluation_criteria}",
                        "render_images": "{render_images}"
                    }
                )
            ])
        
        # 转换为字典格式
        steps_dict = [asdict(step) for step in steps]
        
        return {
            "workflow_id": workflow_id,
            "task_description": task_description,
            "total_steps": len(steps),
            "steps": steps_dict,
            "status": "created",
            "current_step": 0
        }


class ExecuteWorkflowStepTool(LLMTool):
    """执行工作流步骤工具"""
    
    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="execute_workflow_step",
            description="Execute a single step in the workflow",
            parameters=[
                ToolParameter(
                    name="workflow",
                    type="object",
                    description="Current workflow state"
                ),
                ToolParameter(
                    name="step_index",
                    type="integer",
                    description="Index of step to execute"
                ),
                ToolParameter(
                    name="context_data",
                    type="object",
                    description="Data from previous steps",
                    required=False,
                    default={}
                )
            ],
            returns="Dict with updated workflow and step result",
            category="workflow"
        )
    
    def execute(self, workflow: Dict, step_index: int, 
                context_data: Dict = None) -> Dict[str, Any]:
        """执行工作流中的单个步骤"""
        if context_data is None:
            context_data = {}
        
        steps = workflow.get("steps", [])
        
        if step_index >= len(steps):
            return {
                "success": False,
                "error": f"Step index {step_index} out of range"
            }
        
        current_step = steps[step_index]
        step_name = current_step["name"]
        tool_name = current_step["tool_name"]
        parameters = current_step["parameters"].copy()
        
        print(f"执行工作流步骤 {step_index + 1}: {step_name}")
        
        # 替换参数中的占位符
        parameters = self._resolve_placeholders(parameters, context_data)
        
        try:
            # 标记步骤为进行中
            current_step["status"] = "in_progress"
            
            # 执行工具
            if self.tool_registry:
                result = self.tool_registry.execute_function_call(tool_name, parameters)
            else:
                # 模拟执行
                result = {
                    "success": True,
                    "mock": True,
                    "step": step_name,
                    "tool": tool_name
                }
            
            # 更新步骤状态
            if isinstance(result, dict) and result.get("success"):
                current_step["status"] = "completed"
                current_step["result"] = result
                print(f"步骤 {step_index + 1} 完成: {step_name}")
            else:
                current_step["status"] = "failed"
                current_step["error"] = str(result)
                print(f"步骤 {step_index + 1} 失败: {step_name}")
            
            # 更新工作流
            workflow["current_step"] = step_index + 1
            workflow["steps"] = steps
            
            return {
                "success": current_step["status"] == "completed",
                "workflow": workflow,
                "step_result": result,
                "next_step": step_index + 1 if step_index + 1 < len(steps) else None
            }
            
        except Exception as e:
            current_step["status"] = "failed"
            current_step["error"] = str(e)
            
            return {
                "success": False,
                "workflow": workflow,
                "error": str(e),
                "step_name": step_name
            }
    
    def _resolve_placeholders(self, parameters: Dict, context_data: Dict) -> Dict:
        """解析参数中的占位符"""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # 这是一个占位符
                placeholder_key = value[1:-1]  # 移除大括号
                if placeholder_key in context_data:
                    resolved[key] = context_data[placeholder_key]
                else:
                    # 保留占位符，让后续步骤处理
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved


class GetWorkflowStatusTool(LLMTool):
    """获取工作流状态工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_workflow_status",
            description="Get the current status of a workflow",
            parameters=[
                ToolParameter(
                    name="workflow",
                    type="object",
                    description="Workflow object"
                )
            ],
            returns="Dict with workflow status summary",
            category="workflow"
        )
    
    def execute(self, workflow: Dict) -> Dict[str, Any]:
        """获取工作流状态"""
        steps = workflow.get("steps", [])
        total_steps = len(steps)
        
        completed = sum(1 for step in steps if step.get("status") == "completed")
        failed = sum(1 for step in steps if step.get("status") == "failed")
        in_progress = sum(1 for step in steps if step.get("status") == "in_progress")
        pending = sum(1 for step in steps if step.get("status") == "pending")
        
        # 生成状态报告
        status_lines = []
        for i, step in enumerate(steps):
            status = step.get("status", "pending")
            name = step.get("name", f"Step {i+1}")
            
            if status == "completed":
                status_lines.append(f"   Completed Step {i+1}: {name}")
            elif status == "failed":
                error = step.get("error", "Unknown error")
                status_lines.append(f"   Failed Step {i+1}: {name} - {error}")
            elif status == "in_progress":
                status_lines.append(f"   In Progress Step {i+1}: {name}")
            else:
                status_lines.append(f"   Pending Step {i+1}: {name}")
        
        return {
            "workflow_id": workflow.get("workflow_id"),
            "total_steps": total_steps,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "progress_percentage": (completed / total_steps * 100) if total_steps > 0 else 0,
            "status_report": "\n".join(status_lines),
            "is_complete": completed == total_steps,
            "has_failures": failed > 0
        }


def register_workflow_tools(registry, tool_registry=None):
    """
    注册工作流管理工具
    
    Args:
        registry: LLM工具注册中心
        tool_registry: 基础工具注册中心（用于执行实际工具）
    """
    tools = [
        CreateWorkflowTool(),
        ExecuteWorkflowStepTool(tool_registry),
        GetWorkflowStatusTool()
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} workflow management tools\n")
    return tools
