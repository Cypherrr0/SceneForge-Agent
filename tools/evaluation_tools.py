"""
评估和优化相关工具集合
"""
import os
import json
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


class Generate3DEvaluationIndexTool(LLMTool):
    """生成3D评估指标工具"""
    
    def __init__(self, evaluate_agent=None):
        self.agent = evaluate_agent
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="generate_evaluation_index",
            description="Generate evaluation criteria for 3D asset quality assessment",
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description="Original generation prompt"
                ),
                ToolParameter(
                    name="reference_image",
                    type="string",
                    description="Path to reference image or image object"
                )
            ],
            returns="List of evaluation criteria",
            category="evaluation"
        )
    
    def execute(self, prompt: str, reference_image: Union[str, Image.Image]) -> List[str]:
        """生成评估指标"""
        # 检查图像路径是否存在（如果是字符串路径）
        if isinstance(reference_image, str):
            if not os.path.exists(reference_image):
                print(f"⚠️ 参考图像不存在: {reference_image}，使用默认评估指标")
                # 使用默认评估指标
                return [
                    "物体是否存在且完整",
                    "形状匹配度", 
                    "材质和颜色准确性",
                    "纹理细节质量",
                    "整体视觉效果",
                    "多视角一致性",
                    "光照和阴影效果",
                    "模型复杂度适中",
                    "无明显缺陷或错误",
                    "符合提示词要求"
                ]
        
        if self.agent:
            try:
                return self.agent.generate_evaluation_index(prompt, reference_image)
            except Exception as e:
                print(f"⚠️ 评估代理调用失败: {e}，使用默认指标")
                # 如果调用失败，返回默认指标
                return [
                    "物体是否存在且完整",
                    "形状匹配度",
                    "材质和颜色准确性", 
                    "纹理细节质量",
                    "整体视觉效果",
                    "多视角一致性",
                    "光照和阴影效果",
                    "模型复杂度适中",
                    "无明显缺陷或错误",
                    "符合提示词要求"
                ]
        else:
            # 默认评估指标
            return [
                "物体是否存在且完整",
                "形状匹配度",
                "材质和颜色准确性",
                "纹理细节质量", 
                "整体视觉效果",
                "多视角一致性",
                "光照和阴影效果",
                "模型复杂度适中",
                "无明显缺陷或错误",
                "符合提示词要求"
            ]


class Evaluate3DAssetTool(LLMTool):
    """评估3D资产质量工具"""
    
    def __init__(self, evaluate_agent=None):
        self.agent = evaluate_agent
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="evaluate_3d_asset",
            description="Evaluate the quality of generated 3D asset from multiple views",
            parameters=[
                ToolParameter(
                    name="evaluation_index",
                    type="array",
                    description="List of evaluation criteria"
                ),
                ToolParameter(
                    name="render_images",
                    type="array",
                    description="List of rendered image paths or PIL images"
                ),
                ToolParameter(
                    name="threshold",
                    type="integer",
                    description="Score threshold for acceptance",
                    required=False,
                    default=60
                )
            ],
            returns="Dict with score and improvement suggestions",
            category="evaluation"
        )
    
    def execute(self, evaluation_index: List[str], 
                render_images: List[Union[str, Image.Image]],
                threshold: int = 60) -> Dict[str, Any]:
        """评估3D资产"""
        if self.agent:
            score, suggestions = self.agent.evaluate(evaluation_index, render_images)
        else:
            # 模拟评估
            import random
            score = random.randint(40, 90)
            suggestions = "增加更多细节，改善纹理质量" if score < threshold else "质量良好"
        
        return {
            "score": score,
            "passed": score >= threshold,
            "threshold": threshold,
            "improvement_suggestions": suggestions,
            "evaluation_criteria_count": len(evaluation_index),
            "views_evaluated": len(render_images)
        }


class Render3DVideoTool(LLMTool):
    """渲染3D视频工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="render_3d_video",
            description="Render a rotating video of the 3D model",
            parameters=[
                ToolParameter(
                    name="blend_file_path",
                    type="string",
                    description="Path to the Blender file",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="glb_file_path",
                    type="string",
                    description="Path to the GLB model file",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="output_dir",
                    type="string",
                    description="Directory to save the video"
                ),
                ToolParameter(
                    name="uid",
                    type="string",
                    description="Unique identifier",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="fps",
                    type="integer",
                    description="Frames per second",
                    required=False,
                    default=30
                ),
                ToolParameter(
                    name="duration",
                    type="integer",
                    description="Video duration in seconds",
                    required=False,
                    default=10
                )
            ],
            returns="Dict with video_path and render info",
            category="rendering"
        )
    
    def execute(self, output_dir: str, blend_file_path: Optional[str] = None,
                glb_file_path: Optional[str] = None, uid: Optional[str] = None,
                fps: int = 30, duration: int = 10) -> Dict[str, Any]:
        """执行3D视频渲染"""
        if not blend_file_path and not glb_file_path:
            raise ValueError("Either blend_file_path or glb_file_path must be provided")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建视频路径
        video_filename = f"{uid}_rotation.mp4" if uid else "rotation.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        try:
            from blender_tools.render_3D_videos import render_3D_videos
            
            # 调用渲染函数
            if blend_file_path:
                render_3D_videos(save_dir=os.path.dirname(output_dir), uid=uid)
            else:
                # 如果只有GLB，需要先导入到Blender
                print("Note: Direct GLB to video rendering requires Blender setup")
            
            return {
                "success": True,
                "video_path": video_path,
                "fps": fps,
                "duration": duration,
                "total_frames": fps * duration
            }
            
        except ImportError:
            # 模拟结果
            return {
                "success": False,
                "message": "Blender tools not available",
                "mock_result": True,
                "video_path": video_path,
                "fps": fps,
                "duration": duration
            }


class CompareGenerationsTool(LLMTool):
    """比较多个生成结果工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="compare_generations",
            description="Compare multiple 3D generation results",
            parameters=[
                ToolParameter(
                    name="generations",
                    type="array",
                    description="List of generation results with scores"
                ),
                ToolParameter(
                    name="criteria",
                    type="array",
                    description="Comparison criteria",
                    required=False,
                    default=["score", "quality", "accuracy"]
                )
            ],
            returns="Dict with best generation and comparison results",
            category="evaluation"
        )
    
    def execute(self, generations: List[Dict], 
                criteria: List[str] = None) -> Dict[str, Any]:
        """比较多个生成结果"""
        if not criteria:
            criteria = ["score", "quality", "accuracy"]
        
        if not generations:
            return {
                "best": None,
                "comparison": "No generations to compare"
            }
        
        # 按分数排序（假设每个generation都有score字段）
        sorted_gens = sorted(
            generations, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        
        best = sorted_gens[0]
        
        comparison_result = {
            "best": best,
            "best_index": generations.index(best),
            "ranking": [
                {
                    "rank": i + 1,
                    "uid": gen.get("uid", f"gen_{i}"),
                    "score": gen.get("score", 0)
                }
                for i, gen in enumerate(sorted_gens)
            ],
            "total_compared": len(generations),
            "criteria_used": criteria
        }
        
        return comparison_result


class IterativeImprovementTool(LLMTool):
    """迭代改进工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="iterative_improvement",
            description="Manage iterative improvement of 3D generation",
            parameters=[
                ToolParameter(
                    name="current_score",
                    type="number",
                    description="Current generation score"
                ),
                ToolParameter(
                    name="target_score",
                    type="number",
                    description="Target score to achieve"
                ),
                ToolParameter(
                    name="improvement_suggestions",
                    type="string",
                    description="Suggestions for improvement"
                ),
                ToolParameter(
                    name="current_prompt",
                    type="string",
                    description="Current generation prompt"
                ),
                ToolParameter(
                    name="iteration",
                    type="integer",
                    description="Current iteration number"
                ),
                ToolParameter(
                    name="max_iterations",
                    type="integer",
                    description="Maximum iterations allowed",
                    required=False,
                    default=3
                )
            ],
            returns="Dict with decision and improved prompt",
            category="optimization"
        )
    
    def execute(self, current_score: float, target_score: float,
                improvement_suggestions: str, current_prompt: str,
                iteration: int, max_iterations: int = 3) -> Dict[str, Any]:
        """决定是否继续迭代改进"""
        should_continue = (
            current_score < target_score and 
            iteration < max_iterations
        )
        
        improved_prompt = current_prompt
        if should_continue and improvement_suggestions:
            # 将改进建议添加到提示词
            improved_prompt = f"{current_prompt}, {improvement_suggestions}"
        
        return {
            "should_continue": should_continue,
            "reason": (
                "Score below threshold" if current_score < target_score
                else "Target achieved" if current_score >= target_score
                else "Max iterations reached"
            ),
            "improved_prompt": improved_prompt,
            "current_iteration": iteration,
            "max_iterations": max_iterations,
            "score_gap": target_score - current_score,
            "progress_percentage": (current_score / target_score) * 100
        }


def register_evaluation_tools(registry, agents: Optional[Dict] = None):
    """
    注册所有评估相关工具
    
    Args:
        registry: 工具注册中心
        agents: 可选的agent实例字典
    """
    agents = agents or {}
    
    tools = [
        Generate3DEvaluationIndexTool(agents.get("evaluate_agent")),
        Evaluate3DAssetTool(agents.get("evaluate_agent")),
        Render3DVideoTool(),
        CompareGenerationsTool(),
        IterativeImprovementTool()
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} evaluation tools\n")
    return tools
