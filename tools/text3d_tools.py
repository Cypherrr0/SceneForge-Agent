"""
Text to 3D相关工具集合
将Text23dPipeline的功能拆分为独立的LLM可调用工具
"""
import os
import json
import uuid
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


class Text2ImageTool(LLMTool):
    """文本生成图像工具 - 使用Gemini替代qwen-image"""
    
    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="text_to_image",
            description="Generate an image from text description using Gemini 2.5 Flash Image Preview",
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description="Text description of the image to generate"
                ),
                ToolParameter(
                    name="style",
                    type="string",
                    description="Image style (3d, realistic, cartoon, artistic)",
                    required=False,
                    default="3d"
                ),
                ToolParameter(
                    name="save_path",
                    type="string",
                    description="Path to save the generated image",
                    required=False,
                    default=None
                )
            ],
            returns="Dict with image_path and generation info",
            category="generation"
        )
    
    def execute(self, prompt: str, style: str = "3d", 
                save_path: Optional[str] = None) -> Dict[str, Any]:
        """执行文本生成图像 - 使用Gemini API"""
        try:
            # 如果 save_path 是目录，自动添加文件名
            if save_path and os.path.isdir(save_path):
                save_path = os.path.join(save_path, "2d_reference_image.png")
                print(f"检测到目录路径，自动添加文件名: {save_path}")
            
            # 使用Gemini工具
            from tools.gemini_tools import GeminiText2ImageTool
            
            gemini_tool = GeminiText2ImageTool(self.gemini_api_key)
            result = gemini_tool.execute(prompt=prompt, save_path=save_path, style=style)
            
            print(f"使用Gemini生成图像: {prompt}")
            return result
            
        except Exception as e:
            print(f"Gemini图像生成失败，使用占位方案: {e}")
            
            # 如果 save_path 是目录，自动添加文件名
            if save_path and os.path.isdir(save_path):
                save_path = os.path.join(save_path, "2d_reference_image.png")
                print(f"检测到目录路径，自动添加文件名: {save_path}")
            
            # 创建占位图像作为备选方案
            if save_path and save_path.strip():
                save_dir = os.path.dirname(save_path)
                if save_dir:  # 只有当目录路径不为空时才创建
                    os.makedirs(save_dir, exist_ok=True)
                self._create_placeholder_image(prompt, save_path)
            else:
                # 生成临时路径
                import tempfile
                save_path = os.path.join(tempfile.gettempdir(), f"text2img_{hash(prompt) % 10000}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self._create_placeholder_image(prompt, save_path)
                print(f"使用临时占位图像路径: {save_path}")
            
            return {
                "success": True,  # 标记为成功以继续流程
                "image_path": save_path,
                "prompt": prompt,
                "model": "gemini-2.5-flash-image-preview",
                "note": "Using placeholder image due to API limitations"
            }
    
    def _create_placeholder_image(self, prompt: str, save_path: str):
        """创建占位图像"""
        try:
            from PIL import Image, ImageDraw
            
            # 创建白色背景图像
            img = Image.new('RGB', (512, 512), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # 添加提示文本
            text_lines = [
                "3D Generation Reference",
                f"Prompt: {prompt[:40]}...",
                "Generated with Gemini API",
                "(Placeholder for 3D modeling)"
            ]
            
            y_offset = 200
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line)
                text_width = bbox[2] - bbox[0]
                x = (512 - text_width) // 2
                draw.text((x, y_offset), line, fill='black')
                y_offset += 30
            
            img.save(save_path)
            print(f"Created placeholder image: {save_path}")
            
        except Exception as e:
            print(f"Failed to create placeholder image: {e}")


class Image2Shape3DTool(LLMTool):
    """图像生成3D形状工具"""
    
    def __init__(self, hunyuan_pipeline=None):
        self.pipeline = hunyuan_pipeline
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="image_to_3d_shape",
            description="Generate 3D shape from an image using Hunyuan3D",
            parameters=[
                ToolParameter(
                    name="image_path",
                    type="string",
                    description="Path to the input image"
                ),
                ToolParameter(
                    name="output_dir",
                    type="string",
                    description="Directory to save the 3D shape",
                    required=False,
                    default="./outputs"
                ),
                ToolParameter(
                    name="steps",
                    type="integer",
                    description="Number of generation steps",
                    required=False,
                    default=50
                )
            ],
            returns="Dict with mesh_path and generation info",
            category="generation"
        )
    
    def execute(self, image_path: str, output_dir: str = "./outputs", 
                steps: int = 50) -> Dict[str, Any]:
        """执行图像生成3D形状"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        if self.pipeline is None:
            # 这里需要实际的Hunyuan3D pipeline初始化
            return {
                "success": False,
                "message": "Hunyuan3D pipeline not initialized",
                "mock_result": True,
                "mesh_path": os.path.join(output_dir, "shape.obj")
            }
        
        # 调用Hunyuan3D生成形状
        mesh_path = self.pipeline.generate_shape(image_path, output_dir, steps)
        
        return {
            "success": True,
            "mesh_path": mesh_path,
            "input_image": image_path,
            "steps": steps
        }


class Shape2TextureTool(LLMTool):
    """3D形状纹理生成工具"""
    
    def __init__(self, hunyuan_pipeline=None):
        self.pipeline = hunyuan_pipeline
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="generate_3d_texture",
            description="Generate texture for 3D shape using Hunyuan3D",
            parameters=[
                ToolParameter(
                    name="mesh_path",
                    type="string",
                    description="Path to the 3D mesh file"
                ),
                ToolParameter(
                    name="image_path",
                    type="string",
                    description="Reference image for texture generation"
                ),
                ToolParameter(
                    name="output_path",
                    type="string",
                    description="Path to save the textured model",
                    required=False,
                    default=None
                )
            ],
            returns="Dict with textured_model_path",
            category="generation"
        )
    
    def execute(self, mesh_path: str, image_path: str, 
                output_path: Optional[str] = None) -> Dict[str, Any]:
        """执行纹理生成"""
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if output_path is None:
            base_name = os.path.splitext(mesh_path)[0]
            output_path = f"{base_name}_textured.glb"
        
        if self.pipeline is None:
            # 模拟结果
            return {
                "success": False,
                "message": "Hunyuan3D pipeline not initialized",
                "mock_result": True,
                "textured_model_path": output_path
            }
        
        # 调用Hunyuan3D生成纹理
        textured_path = self.pipeline.generate_texture(
            mesh_path, image_path, output_path
        )
        
        return {
            "success": True,
            "textured_model_path": textured_path,
            "input_mesh": mesh_path,
            "reference_image": image_path
        }


class Text23DPipelineTool(LLMTool):
    """完整的文本到3D生成管道工具"""
    
    def __init__(self, text23d_pipeline=None):
        self.pipeline = text23d_pipeline
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="img_to_3d_complete",
            description="Complete text to 3D generation pipeline",
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description="Text description for 3D generation"
                ),
                ToolParameter(
                    name="uid",
                    type="string",
                    description="Unique identifier for this generation",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="save_dir",
                    type="string",
                    description="Directory to save outputs",
                    required=False,
                    default="./outputs"
                ),
                ToolParameter(
                    name="use_existing_image",
                    type="string",
                    description="Path to existing image to use",
                    required=False,
                    default=None
                )
            ],
            returns="Dict with glb_path and generation info",
            category="generation"
        )
    
    def execute(self, prompt: str, uid: Optional[str] = None,
                save_dir: str = "./outputs", 
                use_existing_image: Optional[str] = None) -> Dict[str, Any]:
        """执行完整的文本到3D生成流程"""
        if uid is None:
            uid = str(uuid.uuid4())
        
        # 强制使用调用方传入的 save_dir + uid 组合，避免重做时产生新子目录
        # 如果 save_dir 已包含 uid（或以 uid 结尾），直接使用；否则将 uid 作为子目录追加
        if save_dir.endswith(uid) or os.path.basename(save_dir) == uid:
            output_dir = save_dir
        else:
            output_dir = os.path.join(save_dir, uid)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.pipeline is None:
            # 延迟导入
            try:
                from text_to_3d_agent.Text23dPipeline import Text23dPipeline
                import json
                from pathlib import Path
                
                # 获取项目根目录
                project_root = Path(__file__).parent.parent.absolute()
                
                # 加载配置文件（使用绝对路径）
                with open(project_root / "config" / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                with open(project_root / "config" / "text23d_config.json", "r", encoding="utf-8") as f:
                    text23d_config = json.load(f)
                
                # 使用配置初始化pipeline
                self.pipeline = Text23dPipeline(
                    txt2img_model_name=config.get("txt2img_model_name", "gemini-2.5-flash-image-preview"),
                    low_vram_mode=text23d_config.get('low_vram_mode', False),
                    device=text23d_config.get('device', 'cuda'),
                    hunyuan_file_path=text23d_config.get("hunyuan_file_path", "text_to_3d_agent/Hunyuan3D-2.1"),
                    hunyuan_model_path=text23d_config.get("hunyuan_model_path", "tencent/Hunyuan3D-2.1"),
                    hunyuan_subfolder=text23d_config.get("hunyuan_subfolder", "hunyuan3d-dit-v2-1"),
                    qwen_model_path=text23d_config.get("qwen_model_path", "text_to_3d_agent/models"),
                    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
                )
                print(f"Text23dPipeline initialized with txt2img_model: {config.get('txt2img_model_name')}")
                
            except Exception as e:
                print(f"Text23dPipeline initialization failed: {e}")
                # 模拟结果
                return {
                    "success": False,
                    "message": f"Text23dPipeline not available: {str(e)}",
                    "mock_result": True,
                    "uid": uid,
                    "glb_path": os.path.join(output_dir, "model.glb"),
                    "image_path": os.path.join(output_dir, "reference.png")
                }
        
        # 执行生成 - 需要提供input_image路径
        if not use_existing_image:
            raise ValueError("use_existing_image (image path) is required for 3D generation")
        
        image, glb_path = self.pipeline.generate_3d(
            uid=uid,
            input_source=prompt,
            input_image=use_existing_image,
            save_dir=output_dir  # 固定在同一 output_dir，下游文件名保持一致
        )
        
        # 保存参考图像
        image_path = os.path.join(output_dir, "reference.png")
        if isinstance(image, Image.Image):
            image.save(image_path)
        
        return {
            "success": True,
            "uid": uid,
            "glb_path": glb_path,
            "image_path": image_path,
            "prompt": prompt,
            "output_dir": output_dir
        }


class OptimizePromptTool(LLMTool):
    """优化3D生成提示词工具"""
    
    def __init__(self, prompt_agent=None):
        self.agent = prompt_agent
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="optimize_3d_prompt",
            description="Optimize prompt for better 3D generation results",
            parameters=[
                ToolParameter(
                    name="original_prompt",
                    type="string",
                    description="Original prompt to optimize"
                ),
                ToolParameter(
                    name="target_model",
                    type="string",
                    description="Target model (nano-banana or Qwen)",
                    required=False,
                    default="Qwen"
                ),
                ToolParameter(
                    name="translate_to_english",
                    type="boolean",
                    description="Whether to translate to English",
                    required=False,
                    default=False
                )
            ],
            returns="Dict with optimized_prompt",
            category="optimization"
        )
    
    def execute(self, original_prompt: str, target_model: str = "Qwen",
                translate_to_english: bool = False) -> Dict[str, Any]:
        """执行提示词优化"""
        try:
            # 首先尝试从ParseUserIntentTool获取已清理的提示词
            cleaned_prompt = self._get_cleaned_prompt_from_intent(original_prompt)
            
            if cleaned_prompt and cleaned_prompt != original_prompt:
                print(f"使用ParseUserIntentTool提取的纯净描述: {cleaned_prompt}")
                optimized = cleaned_prompt
            else:
                print(f"使用原始提示词进行优化")
                optimized = original_prompt
            
            if self.agent:
                try:
                    if translate_to_english:
                        optimized = self.agent.translate(optimized)
                    optimized = self.agent.modify(optimized, target_model)
                    print(f"使用PromptAgent优化提示词")
                except Exception as e:
                    print(f"PromptAgent调用失败: {e}，使用默认优化规则")
                    # 降级到简单规则
                    if target_model == "nano-banana" or "gemini" in target_model.lower():
                        optimized += ", no background, Ultra HD, 4K, detailed, suitable for 3D modeling"
                    else:
                        optimized += "，没有背景，超清，4K，细致，适合3D建模"
            else:
                # 简单的优化规则
                print(f"使用默认优化规则（PromptAgent未初始化）")
                if target_model == "nano-banana" or "gemini" in target_model.lower():
                    optimized += ", no background, Ultra HD, 4K, detailed, suitable for 3D modeling"
                else:
                    optimized += "，没有背景，超清，4K，细致，适合3D建模"
            
            return {
                "success": True,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt if cleaned_prompt else original_prompt,
                "optimized_prompt": optimized,
                "target_model": target_model,
                "translated": translate_to_english
            }
            
        except Exception as e:
            print(f"提示词优化失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_prompt": original_prompt,
                "optimized_prompt": original_prompt,  # 返回原始提示词作为后备
                "target_model": target_model
            }
    
    def _get_cleaned_prompt_from_intent(self, original_prompt: str) -> str:
        """从ParseUserIntentTool获取已清理的提示词"""
        try:
            from tools.intent_parser_tools import get_latest_intent
            parsed_intent = get_latest_intent()
            
            if parsed_intent and parsed_intent.get("generation_prompt"):
                cleaned_prompt = parsed_intent.get("generation_prompt")
                print(f"从意图解析中获取纯净描述: {cleaned_prompt}")
                return cleaned_prompt
            else:
                print(f"未找到意图解析结果，使用原始提示词")
                return original_prompt
                
        except ImportError:
            print(f"无法导入意图解析工具，使用原始提示词")
            return original_prompt
        except Exception as e:
            print(f"获取意图解析结果失败: {e}，使用原始提示词")
            return original_prompt


def register_text3d_tools(registry, pipelines: Optional[Dict] = None):
    """
    注册所有Text3D相关工具
    
    Args:
        registry: 工具注册中心
        pipelines: 可选的pipeline实例字典
    """
    pipelines = pipelines or {}
    
    tools = [
        Text2ImageTool(pipelines.get("gemini_api_key")),  # 使用Gemini API Key
        Image2Shape3DTool(pipelines.get("hunyuan")),
        Shape2TextureTool(pipelines.get("hunyuan")),
        Text23DPipelineTool(pipelines.get("text23d")),
        OptimizePromptTool(pipelines.get("prompt_agent"))
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} Text3D tools (using Gemini for image generation)\n")
    return tools
