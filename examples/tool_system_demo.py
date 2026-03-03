#!/usr/bin/env python3
"""
工具管理系统使用示例
演示如何使用新的工具系统进行渲染操作
"""
import os
import sys
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.base import tool_registry
from tools.render_tools import register_all_render_tools


def demo_tool_registry():
    """演示工具注册和管理"""
    print("=" * 60)
    print("工具管理系统演示")
    print("=" * 60)
    
    # 1. 注册所有渲染工具
    print("\n1. 注册渲染工具...")
    register_all_render_tools()
    
    # 2. 列出所有可用工具
    print("\n2. 可用工具列表:")
    all_tools = tool_registry.list_tools()
    for i, tool_name in enumerate(all_tools, 1):
        tool_info = tool_registry.get_tool_info(tool_name)
        print(f"   {i}. {tool_name}: {tool_info['description']}")
    
    # 3. 按类别列出工具
    print("\n3. 按类别分组的工具:")
    categories = tool_registry.list_categories()
    for category in categories:
        tools_in_category = tool_registry.list_tools(category=category)
        print(f"   📁 {category}:")
        for tool_name in tools_in_category:
            print(f"      - {tool_name}")
    
    # 4. 搜索工具
    print("\n4. 搜索工具示例:")
    search_keywords = ["camera", "render", "scene"]
    for keyword in search_keywords:
        results = tool_registry.search_tools(keyword)
        print(f"   搜索 '{keyword}': {results}")
    
    # 5. 获取工具详细信息
    print("\n5. 工具详细信息示例 (render_image):")
    render_tool_info = tool_registry.get_tool_info("render_image")
    print(f"   名称: {render_tool_info['name']}")
    print(f"   描述: {render_tool_info['description']}")
    print(f"   类别: {render_tool_info['category']}")
    print(f"   参数:")
    for param in render_tool_info['parameters']:
        required = "必需" if param['required'] else "可选"
        print(f"      - {param['name']} ({param['type']}, {required}): {param['description']}")
    print(f"   返回: {render_tool_info['returns']}")


def demo_custom_tool():
    """演示如何创建和注册自定义工具"""
    print("\n" + "=" * 60)
    print("自定义工具示例")
    print("=" * 60)
    
    from tools.base import BaseTool, ToolSchema, ToolParameter
    
    class CustomAnalysisTool(BaseTool):
        """自定义分析工具示例"""
        
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(
                name="custom_analysis",
                description="分析3D模型的复杂度",
                parameters=[
                    ToolParameter(
                        name="model_path",
                        type="string",
                        description="模型文件路径"
                    ),
                    ToolParameter(
                        name="detail_level",
                        type="string",
                        description="分析详细程度",
                        required=False,
                        default="medium"
                    )
                ],
                returns="Dict - 分析结果",
                category="analysis"
            )
        
        def execute(self, model_path: str, detail_level: str = "medium") -> dict:
            """执行分析"""
            # 这里是示例实现
            return {
                "model_path": model_path,
                "complexity": "medium",
                "vertices": 10000,
                "faces": 5000,
                "detail_level": detail_level
            }
    
    # 注册自定义工具
    custom_tool = CustomAnalysisTool()
    tool_registry.register(custom_tool)
    
    print("\n✅ 自定义工具已注册")
    
    # 使用自定义工具
    print("\n执行自定义工具:")
    result = tool_registry.execute_tool(
        "custom_analysis",
        model_path="/path/to/model.glb",
        detail_level="high"
    )
    print(f"分析结果: {json.dumps(result, indent=2)}")


def demo_tool_composition():
    """演示如何组合使用多个工具"""
    print("\n" + "=" * 60)
    print("工具组合使用示例")
    print("=" * 60)
    
    class RenderPipeline:
        """使用工具组合的渲染管道"""
        
        def __init__(self):
            self.registry = tool_registry
        
        def process_model(self, glb_path: str, output_dir: str):
            """处理3D模型的完整流程"""
            print(f"\n处理模型: {glb_path}")
            
            # 步骤1: 清空场景
            print("  1. 清空场景...")
            self.registry.execute_tool("clear_scene")
            
            # 步骤2: 导入模型
            print("  2. 导入GLB模型...")
            objects = self.registry.execute_tool("import_glb", filepath=glb_path)
            
            # 步骤3: 获取场景对象
            print("  3. 获取场景对象...")
            mesh_objects = self.registry.execute_tool("get_scene_objects", object_type="MESH")
            
            # 步骤4: 计算边界框
            print("  4. 计算边界框...")
            min_corner, max_corner = self.registry.execute_tool(
                "calculate_bounding_box",
                objects=mesh_objects
            )
            
            # 步骤5: 定位到地面
            print("  5. 将对象定位到地面...")
            self.registry.execute_tool("position_on_ground", objects=mesh_objects)
            
            print("\n✅ 模型处理完成!")
            return {
                "objects_count": len(mesh_objects),
                "bounding_box": {
                    "min": str(min_corner),
                    "max": str(max_corner)
                }
            }
    
    # 创建管道实例
    pipeline = RenderPipeline()
    
    # 示例：处理模型（这里使用虚拟路径）
    print("\n使用工具组合处理模型:")
    # result = pipeline.process_model("/path/to/model.glb", "/path/to/output")
    print("（示例代码，需要实际的GLB文件路径才能运行）")


def main():
    """主函数"""
    print("\n🔧 工具管理系统演示程序\n")
    
    # 运行各个演示
    demo_tool_registry()
    demo_custom_tool()
    demo_tool_composition()
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("工具系统统计")
    print("=" * 60)
    
    all_info = tool_registry.get_all_tools_info()
    print(f"总工具数: {all_info['total_tools']}")
    print(f"工具类别: {len(all_info['categories'])}")
    for category, tools in all_info['categories'].items():
        print(f"  - {category}: {len(tools)} 个工具")
    
    print("\n✅ 演示完成!")


if __name__ == "__main__":
    main()
