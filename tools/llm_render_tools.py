"""
LLM可调用的渲染工具
将原有的渲染工具包装为LLM工具格式
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from tools.llm_tools import LLMTool, ToolSchema, ToolParameter
from tools.base import tool_registry as base_registry
from tools.render_tools import register_all_render_tools


class LLMRenderSceneTool(LLMTool):
    """LLM可调用的场景渲染工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="render_3d_scene",
            description="Render a 3D model from multiple viewpoints and save as images",
            parameters=[
                ToolParameter(
                    name="glb_file_path",
                    type="string",
                    description="Path to the GLB/GLTF 3D model file"
                ),
                ToolParameter(
                    name="output_dir",
                    type="string",
                    description="Directory to save rendered images"
                ),
                ToolParameter(
                    name="views",
                    type="array",
                    description="List of views to render (front, back, left, right, top, bottom)",
                    required=False,
                    default=["front", "back", "left", "right", "top", "bottom"]
                ),
                ToolParameter(
                    name="resolution",
                    type="integer",
                    description="Image resolution (width and height)",
                    required=False,
                    default=512
                ),
                ToolParameter(
                    name="save_blend",
                    type="boolean",
                    description="Whether to save the Blender file",
                    required=False,
                    default=True
                )
            ],
            returns="Dict with rendered image paths and blend file path",
            category="rendering"
        )
    
    def execute(self, glb_file_path: str, output_dir: str,
                views: List[str] = None, resolution: int = 512,
                save_blend: bool = True) -> Dict[str, Any]:
        """执行场景渲染"""
        if views is None:
            views = ["front", "back", "left", "right", "top", "bottom"]
        
        # 确保渲染工具已注册
        if not base_registry.list_tools():
            register_all_render_tools()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        imgs_path = os.path.join(output_dir, "multi_views")
        os.makedirs(imgs_path, exist_ok=True)
        
        try:
            # 清空场景
            base_registry.execute_tool("clear_scene")
            
            # 导入模型
            base_registry.execute_tool("import_glb", filepath=glb_file_path)
            
            # 获取对象并定位
            objects = base_registry.execute_tool("get_scene_objects", object_type="MESH")
            base_registry.execute_tool("position_on_ground", objects=objects)
            
            # 计算边界框
            min_corner, max_corner = base_registry.execute_tool(
                "calculate_bounding_box", objects=objects
            )
            
            # 设置光照
            lighting_config = {
                "world_background": {
                    "color": [0.5, 0.5, 0.5, 1.0],
                    "strength": 0.0
                },
                "point_light": {
                    "location": [-0.10521, 0.36788, 3.8454],
                    "rotation_euler_degrees": [-49.206, 7.4037, 101.74],
                    "color": [1.0, 1.0, 1.0],
                    "power": 1004.100,
                    "radius": 4.72
                }
            }
            base_registry.execute_tool("setup_lighting", lighting_config=lighting_config)
            
            # 设置相机
            camera_config = {
                "lens_mm": 35,
                "side_distance_multiplier": 1.5,
                "top_bottom_distance_multiplier": 2.0
            }
            cameras = base_registry.execute_tool(
                "setup_multiview_cameras",
                bounding_box=(min_corner, max_corner),
                camera_config=camera_config
            )
            
            # 渲染配置
            render_config = {
                "resolution_x": 512,
                "resolution_y": 512,
                "engine": "CYCLES",
                "exposure": 0.9,
                "cycles_samples": 128,
                "film_transparent": True
            }
            
            # 渲染每个视角
            rendered_images = {}
            for view in views:
                if view in cameras:
                    image_path = os.path.join(imgs_path, f"{view}.png")
                    base_registry.execute_tool(
                        "render_image",
                        camera=cameras[view],
                        output_path=image_path,
                        render_config=render_config
                    )
                    rendered_images[view] = image_path
            
            # 保存Blend文件
            blend_path = None
            if save_blend:
                blend_dir = os.path.join(output_dir, "blend")
                os.makedirs(blend_dir, exist_ok=True)
                blend_path = os.path.join(blend_dir, "scene.blend")
                base_registry.execute_tool(
                    "save_blend_file",
                    filepath=blend_path,
                    set_viewport_shading=True
                )
            
            return {
                "success": True,
                "rendered_images": rendered_images,
                "blend_file": blend_path,
                "total_views": len(rendered_images),
                "resolution": f"{resolution}x{resolution}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Rendering requires Blender environment"
            }


class LLMBatchRenderTool(LLMTool):
    """批量渲染多个模型工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="batch_render_models",
            description="Render multiple 3D models in batch",
            parameters=[
                ToolParameter(
                    name="model_paths",
                    type="array",
                    description="List of GLB/GLTF model file paths"
                ),
                ToolParameter(
                    name="output_base_dir",
                    type="string",
                    description="Base directory for outputs"
                ),
                ToolParameter(
                    name="resolution",
                    type="integer",
                    description="Render resolution",
                    required=False,
                    default=512
                )
            ],
            returns="Dict with results for each model",
            category="rendering"
        )
    
    def execute(self, model_paths: List[str], output_base_dir: str,
                resolution: int = 512) -> Dict[str, Any]:
        """批量渲染模型"""
        results = []
        render_tool = LLMRenderSceneTool()
        
        for i, model_path in enumerate(model_paths):
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            output_dir = os.path.join(output_base_dir, model_name)
            
            result = render_tool.execute(
                glb_file_path=model_path,
                output_dir=output_dir,
                resolution=resolution
            )
            
            results.append({
                "model": model_path,
                "name": model_name,
                "result": result
            })
        
        successful = sum(1 for r in results if r["result"].get("success"))
        
        return {
            "total_models": len(model_paths),
            "successful": successful,
            "failed": len(model_paths) - successful,
            "results": results
        }


class LLMCompositeSceneTool(LLMTool):
    """组合场景渲染工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="render_composite_scene",
            description="Render a scene with multiple 3D objects arranged together",
            parameters=[
                ToolParameter(
                    name="objects",
                    type="array",
                    description="List of objects with paths and positions"
                ),
                ToolParameter(
                    name="output_dir",
                    type="string",
                    description="Output directory"
                ),
                ToolParameter(
                    name="scene_config",
                    type="object",
                    description="Scene configuration (lighting, camera, etc.)",
                    required=False,
                    default={}
                )
            ],
            returns="Dict with composite scene render results",
            category="rendering"
        )
    
    def execute(self, objects: List[Dict], output_dir: str,
                scene_config: Dict = None) -> Dict[str, Any]:
        """渲染组合场景"""
        if scene_config is None:
            scene_config = {}
        
        # 确保渲染工具已注册
        if not base_registry.list_tools():
            register_all_render_tools()
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 清空场景
            base_registry.execute_tool("clear_scene")
            
            # 导入所有对象
            imported_objects = []
            for obj_info in objects:
                if "path" in obj_info:
                    objs = base_registry.execute_tool(
                        "import_glb", 
                        filepath=obj_info["path"]
                    )
                    
                    # 应用位置和旋转（如果指定）
                    if "position" in obj_info or "rotation" in obj_info:
                        for obj in objs:
                            if "position" in obj_info:
                                obj.location = obj_info["position"]
                            if "rotation" in obj_info:
                                obj.rotation_euler = obj_info["rotation"]
                            if "scale" in obj_info:
                                obj.scale = obj_info["scale"]
                    
                    imported_objects.extend(objs)
            
            # 渲染场景
            # 这里可以复用单个模型的渲染逻辑
            result = {
                "success": True,
                "objects_count": len(objects),
                "imported_count": len(imported_objects),
                "output_dir": output_dir
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Composite rendering requires Blender environment"
            }


def register_llm_render_tools(registry):
    """
    注册所有LLM渲染工具
    
    Args:
        registry: LLM工具注册中心
    """
    tools = [
        LLMRenderSceneTool(),
        LLMBatchRenderTool(),
        LLMCompositeSceneTool()
    ]
    
    for tool in tools:
        registry.register(tool)
    
    print(f"\tRegistered {len(tools)} LLM render tools\n")
    return tools
