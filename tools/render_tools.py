"""
渲染相关的工具集合
将原render_agent.py的功能拆分为独立的工具
"""
import bpy
import os
import math
import json
from PIL import Image
from mathutils import Vector
from typing import List, Tuple, Dict, Any, Optional

from tools.base import BaseTool, ToolSchema, ToolParameter, register_tool


class SceneClearTool(BaseTool):
    """清空场景工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="clear_scene",
            description="删除当前Blender场景中的所有对象",
            parameters=[],
            returns="None",
            category="scene_management"
        )
    
    def execute(self) -> None:
        """清空场景中的所有对象"""
        # 确保处于对象模式
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # 方法1：通过bpy.data直接删除所有对象（更彻底）
        # 删除所有网格对象
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # 清理孤立的数据块
        for mesh in list(bpy.data.meshes):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        
        for material in list(bpy.data.materials):
            if material.users == 0:
                bpy.data.materials.remove(material)
        
        for camera in list(bpy.data.cameras):
            if camera.users == 0:
                bpy.data.cameras.remove(camera)
        
        for light in list(bpy.data.lights):
            if light.users == 0:
                bpy.data.lights.remove(light)
        
        print("Scene cleared successfully")


class ImportGLBTool(BaseTool):
    """导入GLB模型工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="import_glb",
            description="导入GLB/GLTF格式的3D模型文件",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="GLB/GLTF文件路径"
                )
            ],
            returns="List[bpy.types.Object] - 导入的对象列表",
            category="import_export"
        )
    
    def execute(self, filepath: str) -> List:
        """导入GLB文件"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 记录导入前的对象
        before_import = set(bpy.context.scene.objects)
        
        # 导入GLB文件
        bpy.ops.import_scene.gltf(filepath=filepath)
        
        # 获取新导入的对象
        after_import = set(bpy.context.scene.objects)
        imported_objects = list(after_import - before_import)
        
        print(f"Imported {len(imported_objects)} objects from {filepath}")
        return imported_objects


class GetSceneObjectsTool(BaseTool):
    """获取场景对象工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_scene_objects",
            description="获取场景中所有的网格对象",
            parameters=[
                ToolParameter(
                    name="object_type",
                    type="string",
                    description="对象类型过滤器（MESH, CAMERA, LIGHT等）",
                    required=False,
                    default="MESH"
                )
            ],
            returns="List[bpy.types.Object] - 场景中的对象列表",
            category="scene_management"
        )
    
    def execute(self, object_type: str = "MESH") -> List:
        """获取场景中的对象"""
        objects = [obj for obj in bpy.context.scene.objects if obj.type == object_type]
        print(f"Found {len(objects)} {object_type} objects in scene")
        return objects


class CalculateBoundingBoxTool(BaseTool):
    """计算边界框工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculate_bounding_box",
            description="计算对象组合的边界框",
            parameters=[
                ToolParameter(
                    name="objects",
                    type="List[bpy.types.Object]",
                    description="要计算边界框的对象列表"
                )
            ],
            returns="Tuple[Vector, Vector] - (最小角点, 最大角点)",
            category="geometry"
        )
    
    def execute(self, objects: List) -> Tuple[Vector, Vector]:
        """计算对象组合的边界框"""
        if not objects:
            return Vector((0, 0, 0)), Vector((0, 0, 0))
        
        min_corner = Vector((float('inf'), float('inf'), float('inf')))
        max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for obj in objects:
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_corner.x = min(min_corner.x, world_corner.x)
                min_corner.y = min(min_corner.y, world_corner.y)
                min_corner.z = min(min_corner.z, world_corner.z)
                max_corner.x = max(max_corner.x, world_corner.x)
                max_corner.y = max(max_corner.y, world_corner.y)
                max_corner.z = max(max_corner.z, world_corner.z)
        
        print(f"Bounding box calculated: min={min_corner}, max={max_corner}")
        return min_corner, max_corner


class PositionObjectsOnGroundTool(BaseTool):
    """将对象放置在地面工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="position_on_ground",
            description="将对象组合移动到世界原点并放置在地面（Z=0）上",
            parameters=[
                ToolParameter(
                    name="objects",
                    type="List[bpy.types.Object]",
                    description="要定位的对象列表"
                )
            ],
            returns="Vector - 应用的平移向量",
            category="transform"
        )
    
    def execute(self, objects: List) -> Vector:
        """将对象放置在地面上"""
        if not objects:
            return Vector((0, 0, 0))
        
        # 计算边界框
        bbox_tool = CalculateBoundingBoxTool()
        min_corner, max_corner = bbox_tool.execute(objects=objects)
        
        # 计算平移向量
        center_x = (min_corner.x + max_corner.x) / 2.0
        center_y = (min_corner.y + max_corner.y) / 2.0
        bottom_z = min_corner.z
        
        translation_vector = Vector((-center_x, -center_y, -bottom_z))
        
        # 直接修改对象位置（更可靠，适用于后台模式）
        for obj in objects:
            obj.location += translation_vector
        
        print(f"Objects positioned on ground with translation: {translation_vector}")
        return translation_vector


class SetupCameraTool(BaseTool):
    """设置相机工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="setup_camera",
            description="创建并配置相机",
            parameters=[
                ToolParameter(
                    name="location",
                    type="Tuple[float, float, float]",
                    description="相机位置"
                ),
                ToolParameter(
                    name="rotation",
                    type="Tuple[float, float, float]",
                    description="相机旋转（弧度）",
                    required=False,
                    default=(0, 0, 0)
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="相机名称"
                ),
                ToolParameter(
                    name="lens",
                    type="float",
                    description="焦距（mm）",
                    required=False,
                    default=35
                )
            ],
            returns="bpy.types.Object - 创建的相机对象",
            category="camera"
        )
    
    def execute(self, location: Tuple[float, float, float], 
                rotation: Tuple[float, float, float] = (0, 0, 0),
                name: str = "Camera", 
                lens: float = 35) -> Any:
        """创建并配置相机"""
        bpy.ops.object.camera_add(location=location, rotation=rotation)
        # 在后台模式下使用 active_object 更可靠
        camera = bpy.context.view_layer.objects.active
        if camera is None:
            camera = bpy.context.active_object
        camera.name = name
        camera.data.lens = lens
        
        print(f"Camera '{name}' created at {location}")
        return camera


class LookAtTool(BaseTool):
    """相机朝向工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="look_at",
            description="使对象朝向目标点",
            parameters=[
                ToolParameter(
                    name="obj",
                    type="bpy.types.Object",
                    description="要旋转的对象"
                ),
                ToolParameter(
                    name="target_point",
                    type="Tuple[float, float, float]",
                    description="目标点坐标"
                )
            ],
            returns="None",
            category="transform"
        )
    
    def execute(self, obj: Any, target_point: Tuple[float, float, float]) -> None:
        """使对象朝向目标点"""
        target = Vector(target_point)
        direction = target - obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        obj.rotation_euler = rot_quat.to_euler()
        
        print(f"Object '{obj.name}' now looking at {target_point}")


class SetupLightingTool(BaseTool):
    """设置光照工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="setup_lighting",
            description="设置场景光照",
            parameters=[
                ToolParameter(
                    name="lighting_config",
                    type="dict",
                    description="光照配置字典"
                )
            ],
            returns="bpy.types.Object - 创建的光源对象",
            category="lighting"
        )
    
    def execute(self, lighting_config: Dict[str, Any]) -> Any:
        """设置光照"""
        # 设置世界背景
        world_bg_config = lighting_config.get("world_background", {})
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = tuple(
                world_bg_config.get("color", [0.5, 0.5, 0.5, 1.0])
            )
            bg_node.inputs['Strength'].default_value = world_bg_config.get("strength", 0.0)
        
        # 添加点光源
        pl_config = lighting_config.get("point_light", {})
        bpy.ops.object.light_add(
            type='POINT', 
            align='WORLD', 
            location=tuple(pl_config.get("location", [0, 0, 0]))
        )
        # 在后台模式下使用 active_object 更可靠
        point_light = bpy.context.view_layer.objects.active
        if point_light is None:
            point_light = bpy.context.active_object
        
        # 设置光源属性
        rotation_degrees = pl_config.get("rotation_euler_degrees", [0, 0, 0])
        point_light.rotation_euler = tuple(math.radians(deg) for deg in rotation_degrees)
        point_light.data.color = tuple(pl_config.get("color", [1, 1, 1]))
        point_light.data.energy = pl_config.get("power", 1000)
        point_light.data.shadow_soft_size = pl_config.get("radius", 0.1)
        point_light.data.use_shadow = True
        
        print("Lighting setup completed")
        return point_light


class SetupMultiViewCamerasTool(BaseTool):
    """设置多视角相机工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="setup_multiview_cameras",
            description="创建六个视角的相机（前后左右上下）",
            parameters=[
                ToolParameter(
                    name="bounding_box",
                    type="Tuple[Vector, Vector]",
                    description="场景边界框（min_corner, max_corner）"
                ),
                ToolParameter(
                    name="camera_config",
                    type="dict",
                    description="相机配置"
                )
            ],
            returns="Dict[str, bpy.types.Object] - 相机字典",
            category="camera"
        )
    
    def execute(self, bounding_box: Tuple[Vector, Vector], 
                camera_config: Dict[str, Any]) -> Dict[str, Any]:
        """设置多视角相机"""
        min_corner, max_corner = bounding_box
        dimensions = max_corner - min_corner
        max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
        
        # 创建临时相机获取默认角度
        temp_cam_data = bpy.data.cameras.new("TempCam")
        temp_cam_data.lens = camera_config.get("lens_mm", 35)
        camera_angle = temp_cam_data.angle
        bpy.data.cameras.remove(temp_cam_data)
        
        # 计算相机距离
        base_distance = max_dimension / (2.0 * math.tan(camera_angle / 2.0))
        side_distance = base_distance * camera_config.get("side_distance_multiplier", 1.5)
        top_bottom_distance = base_distance * camera_config.get("top_bottom_distance_multiplier", 1.5)
        
        # 定义相机位置
        camera_positions = {
            'front': (0, -side_distance, dimensions.z / 2),
            'back': (0, side_distance, dimensions.z / 2),
            'right': (side_distance, 0, dimensions.z / 2),
            'left': (-side_distance, 0, dimensions.z / 2),
            'top': (0, 0, max_corner.z + top_bottom_distance),
            'bottom': (0, 0, min_corner.z - top_bottom_distance)
        }
        
        cameras = {}
        target_point = Vector((0, 0, dimensions.z / 2))
        setup_camera_tool = SetupCameraTool()
        look_at_tool = LookAtTool()
        
        for name, pos in camera_positions.items():
            cam = setup_camera_tool.execute(
                location=pos,
                name=name,
                lens=camera_config.get("lens_mm", 35)
            )
            
            if name not in ['top', 'bottom']:
                look_at_tool.execute(obj=cam, target_point=tuple(target_point))
            else:
                look_at_tool.execute(obj=cam, target_point=(0, 0, 0))
            
            cameras[name] = cam
        
        # 特殊处理顶部和底部相机
        if 'top' in cameras:
            cameras['top'].rotation_euler = (0, 0, 0)
        if 'bottom' in cameras:
            cameras['bottom'].rotation_euler = (math.radians(180), 0, 0)
        
        print(f"Created {len(cameras)} multi-view cameras")
        return cameras


class RenderImageTool(BaseTool):
    """渲染图像工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="render_image",
            description="使用指定相机渲染图像",
            parameters=[
                ToolParameter(
                    name="camera",
                    type="bpy.types.Object",
                    description="用于渲染的相机"
                ),
                ToolParameter(
                    name="output_path",
                    type="string",
                    description="输出图像路径"
                ),
                ToolParameter(
                    name="render_config",
                    type="dict",
                    description="渲染配置",
                    required=False,
                    default={}
                )
            ],
            returns="PIL.Image - 渲染的图像",
            category="rendering"
        )
    
    def execute(self, camera: Any, output_path: str, 
                render_config: Dict[str, Any] = None) -> Image.Image:
        """渲染图像"""
        if render_config is None:
            render_config = {}
        
        scene = bpy.context.scene
        
        # 设置渲染参数
        scene.render.image_settings.file_format = 'PNG'
        scene.render.resolution_x = render_config.get("resolution_x", 512)
        scene.render.resolution_y = render_config.get("resolution_y", 512)
        scene.render.engine = render_config.get("engine", "CYCLES")
        scene.view_settings.exposure = render_config.get("exposure", 0.0)
        scene.cycles.samples = render_config.get("cycles_samples", 128)
        scene.render.film_transparent = render_config.get("film_transparent", True)
        
        # 设置相机
        scene.camera = camera
        
        # 设置输出路径
        scene.render.filepath = output_path
        
        # 渲染
        bpy.ops.render.render(write_still=True)
        
        # 加载并返回图像
        img = Image.open(output_path).convert("RGB")
        print(f"Rendered image saved to {output_path}")
        
        return img


class SaveBlendFileTool(BaseTool):
    """保存Blender文件工具"""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="save_blend_file",
            description="保存当前场景为.blend文件",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="保存路径"
                ),
                ToolParameter(
                    name="set_viewport_shading",
                    type="bool",
                    description="是否设置视口着色为材质预览",
                    required=False,
                    default=True
                )
            ],
            returns="string - 保存的文件路径",
            category="import_export"
        )
    
    def execute(self, filepath: str, set_viewport_shading: bool = True) -> str:
        """保存Blender文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 设置视口着色
        if set_viewport_shading:
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.shading.type = 'MATERIAL'
        
        # 保存文件
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        print(f"Blend file saved to {filepath}")
        
        return filepath


# 注册所有工具
def register_all_render_tools():
    """注册所有渲染相关工具"""
    from tools.base import tool_registry
    
    tools = [
        SceneClearTool(),
        ImportGLBTool(),
        GetSceneObjectsTool(),
        CalculateBoundingBoxTool(),
        PositionObjectsOnGroundTool(),
        SetupCameraTool(),
        LookAtTool(),
        SetupLightingTool(),
        SetupMultiViewCamerasTool(),
        RenderImageTool(),
        SaveBlendFileTool()
    ]
    
    for tool in tools:
        tool_registry.register(tool)
    
    print(f"\tRegistered {len(tools)} render tools\n")
    return tool_registry
