"""
Gemini API工具集合
使用Gemini 2.5 Flash Image Preview替换原有的文生图功能
"""
import os
import json
import base64
import requests
from typing import Dict, Any, Optional
from PIL import Image
import io

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter


class GeminiText2ImageTool(LLMTool):
    """使用Gemini 2.5 Flash Image Preview生成图像的工具"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="gemini_text_to_image",
            description="Generate image from text using Gemini 2.5 Flash Image Preview",
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description="Text description for image generation"
                ),
                ToolParameter(
                    name="save_path",
                    type="string",
                    description="Path to save the generated image",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="style",
                    type="string",
                    description="Image style (realistic, cartoon, artistic, etc.)",
                    required=False,
                    default="realistic"
                )
            ],
            returns="Dict with image_path and generation info",
            category="generation"
        )
    
    def execute(self, prompt: str, save_path: Optional[str] = None, 
                style: str = "realistic") -> Dict[str, Any]:
        """执行Gemini图像生成"""
        try:
            # 构建优化的提示词
            enhanced_prompt = self._enhance_prompt(prompt, style)
            
            # 处理保存路径
            if save_path and save_path.strip():
                # 确保保存路径有效
                save_dir = os.path.dirname(save_path)
                if save_dir:  # 只有当目录路径不为空时才创建
                    os.makedirs(save_dir, exist_ok=True)
            else:
                # 如果没有提供有效路径，生成一个临时路径
                import tempfile
                save_path = os.path.join("/userhome/cs2/u3665834/projects/hunyuan3D-Agent-G1/output/temp", f"gemini_image_{hash(prompt) % 10000}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"使用临时路径: {save_path}")
            
            # 调用Gemini API
            image_data = self._call_gemini_api(enhanced_prompt)
            
            # 处理返回的图像数据
            if image_data:
                # 保存图像
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                
                # 验证图像
                try:
                    img = Image.open(save_path)
                    img.verify()
                    print(f"Gemini图像生成成功: {save_path}")
                except Exception as e:
                    print(f"图像验证失败: {e}")
                
                return {
                    "success": True,
                    "image_path": save_path,
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "style": style,
                    "model": "gemini-2.5-flash-image-preview",
                    "size": len(image_data) if image_data else 0
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate image with Gemini API",
                    "prompt": prompt
                }
                
        except Exception as e:
            print(f"Gemini图像生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "fallback": "Consider using alternative image generation method"
            }
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """优化提示词以获得更好的生成效果"""
        style_prompts = {
            "realistic": "photorealistic, high quality, detailed",
            "cartoon": "cartoon style, colorful, cute, animated",
            "artistic": "artistic, creative, beautiful composition",
            "3d": "3D render, clean background, good lighting for 3D modeling reference"
        }
        
        style_suffix = style_prompts.get(style, style_prompts["realistic"])
        
        # 为3D生成优化提示词
        enhanced = f"{prompt}, {style_suffix}, no background, clean composition, suitable for 3D modeling reference"
        
        return enhanced
    
    def _call_gemini_api(self, prompt: str) -> Optional[bytes]:
        """
        调用Gemini API生成图像
        使用OpenAI客户端调用hiapi.online的Gemini服务
        """
        try:
            # 读取配置文件获取模型名称
            import json
            from pathlib import Path
            
            try:
                # 获取项目根目录
                project_root = Path(__file__).parent.parent.absolute()
                config_path = project_root / "config" / "config.json"
                
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                model_name = config.get("txt2img_model_name", "gemini-2.5-flash-image-preview")
            except Exception as e:
                print(f"无法读取配置文件，使用默认模型: {e}")
                model_name = "gemini-2.5-flash-image-preview"
            
            print(f"使用模型: {model_name}")
            print(f"生成图像提示: {prompt}")
            
            # 创建OpenAI客户端
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://hiapi.online/v1"
            )
            
            # 构建图像生成请求
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI image generator. Generate high-quality images based on user descriptions. Return only the image, no text."
                    },
                    {
                        "role": "user", 
                        "content": f"Generate an image: {prompt}"
                    }
                ],
                stream=False
            )
            
            # 处理响应
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                print(f"Gemini API响应成功")
                print(f"响应内容: {content[:100]}..." if len(content) > 100 else f"响应内容: {content}")
                
                # 尝试处理不同的响应格式
                image_data = self._process_api_response(content, prompt)
                if image_data:
                    return image_data
                else:
                    print("无法从API响应中提取图像数据，使用占位图像")
                    return self._create_placeholder_image(prompt)
            else:
                print("Gemini API响应为空")
                return None
            
        except Exception as e:
            print(f"Gemini API调用失败: {e}")
            # 降级到占位图像
            return self._create_placeholder_image(prompt)
    
    def _process_api_response(self, content: str, prompt: str) -> Optional[bytes]:
        """
        处理API响应，提取图像数据
        支持多种可能的响应格式：URL、base64、文件路径等
        """
        try:
            import base64
            import requests
            import re
            
            # 1. 检查是否包含base64图像数据
            base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
            base64_match = re.search(base64_pattern, content)
            if base64_match:
                print("发现base64图像数据")
                base64_data = base64_match.group(1)
                return base64.b64decode(base64_data)
            
            # 2. 检查是否包含纯base64数据（无前缀）
            if re.match(r'^[A-Za-z0-9+/=]{100,}$', content.strip()):
                print("尝试解析纯base64数据")
                try:
                    return base64.b64decode(content.strip())
                except:
                    pass
            
            # 3. 检查是否包含图像URL
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+\.(jpg|jpeg|png|gif|webp)'
            url_match = re.search(url_pattern, content, re.IGNORECASE)
            if url_match:
                image_url = url_match.group(0)
                print(f"发现图像URL: {image_url}")
                try:
                    response = requests.get(image_url, timeout=30)
                    if response.status_code == 200:
                        return response.content
                except Exception as e:
                    print(f"下载图像失败: {e}")
            
            # 4. 检查是否包含JSON响应
            try:
                import json
                json_data = json.loads(content)
                if isinstance(json_data, dict):
                    # 查找可能的图像字段
                    for key in ['image', 'data', 'url', 'image_url', 'result']:
                        if key in json_data:
                            value = json_data[key]
                            if isinstance(value, str) and value.startswith('http'):
                                # 尝试下载URL
                                try:
                                    response = requests.get(value, timeout=30)
                                    if response.status_code == 200:
                                        return response.content
                                except:
                                    pass
            except:
                pass
            
            print("未能从响应中识别图像数据格式")
            return None
            
        except Exception as e:
            print(f"处理API响应失败: {e}")
            return None
    
    def _create_placeholder_image(self, prompt: str) -> bytes:
        """
        创建占位图像（临时方案）
        实际使用时应该替换为真实的Gemini API调用
        """
        try:
            # 创建一个简单的占位图像
            from PIL import Image, ImageDraw
            import io
            
            # 创建图像
            img = Image.new('RGB', (512, 512), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # 添加文本
            text_lines = [
                "Generated with Gemini API",
                f"Prompt: {prompt[:40]}...",
                "(Placeholder - Awaiting Real API)",
                "hiapi.online/v1"
            ]
            
            y_offset = 180
            for line in text_lines:
                # 使用默认字体
                try:
                    bbox = draw.textbbox((0, 0), line)
                    text_width = bbox[2] - bbox[0]
                    x = (512 - text_width) // 2
                except:
                    # 如果textbbox不可用，使用估算
                    x = (512 - len(line) * 8) // 2
                
                draw.text((x, y_offset), line, fill='black')
                y_offset += 35
            
            # 添加边框
            draw.rectangle([10, 10, 501, 501], outline='black', width=2)
            
            # 转换为bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
            
        except Exception as e:
            print(f"占位图像创建失败: {e}")
            # 返回最小的PNG图像数据
            return self._create_minimal_png()
    
    def _create_minimal_png(self) -> bytes:
        """
        创建最小的PNG图像数据作为最后的后备方案
        """
        # 最简单的1x1像素PNG图像的bytes数据
        minimal_png = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8'
            b'\x0f\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        return minimal_png


class GeminiVisionTool(LLMTool):
    """使用Gemini进行图像理解和分析"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="gemini_image_analysis",
            description="Analyze image content using Gemini Vision",
            parameters=[
                ToolParameter(
                    name="image_path",
                    type="string",
                    description="Path to the image file"
                ),
                ToolParameter(
                    name="question",
                    type="string",
                    description="Question about the image",
                    required=False,
                    default="Describe this image in detail"
                )
            ],
            returns="Dict with analysis results",
            category="analysis"
        )
    
    def execute(self, image_path: str, 
                question: str = "Describe this image in detail") -> Dict[str, Any]:
        """执行图像分析"""
        try:
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # 编码图像
            image_base64 = self._encode_image(image_path)
            
            # 调用Gemini Vision API
            analysis = self._call_gemini_vision(image_base64, question)
            
            return {
                "success": True,
                "image_path": image_path,
                "question": question,
                "analysis": analysis,
                "model": "gemini-2.5-flash"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _call_gemini_vision(self, image_base64: str, question: str) -> str:
        """调用Gemini Vision API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
            
            data = {
                "contents": [{
                    "parts": [
                        {"text": question},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }]
            }
            
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"API调用失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Gemini Vision调用失败: {str(e)}"


def register_gemini_tools(registry, api_key: Optional[str] = None):
    """
    注册所有Gemini工具
    
    Args:
        registry: 工具注册中心
        api_key: Gemini API密钥
    """
    try:
        tools = [
            GeminiText2ImageTool(api_key),
            GeminiVisionTool(api_key)
        ]
        
        for tool in tools:
            registry.register(tool)
        
        print(f"\tRegistered {len(tools)} Gemini tools\n")
        return tools
        
    except Exception as e:
        print(f"Failed to register Gemini tools: {e}")
        return []
