"""
工具系统基础类和接口定义
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import json
import inspect


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    """工具模式定义"""
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str
    category: str = "general"


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self):
        self._schema = None
        self._validate_schema()
    
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """返回工具的schema定义"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具的主要逻辑"""
        pass
    
    def _validate_schema(self):
        """验证schema是否正确定义"""
        if not isinstance(self.schema, ToolSchema):
            raise ValueError(f"Tool {self.__class__.__name__} must define a valid ToolSchema")
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """验证输入参数"""
        validated = {}
        
        for param in self.schema.parameters:
            if param.required and param.name not in kwargs:
                if param.default is not None:
                    validated[param.name] = param.default
                else:
                    raise ValueError(f"Required parameter '{param.name}' is missing")
            elif param.name in kwargs:
                validated[param.name] = kwargs[param.name]
                # 这里可以添加类型检查
        
        return validated
    
    def __call__(self, **kwargs) -> Any:
        """使工具可以直接调用"""
        validated_params = self.validate_parameters(**kwargs)
        return self.execute(**validated_params)
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "category": self.schema.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in self.schema.parameters
            ],
            "returns": self.schema.returns
        }


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: BaseTool, override: bool = False) -> None:
        """注册工具"""
        tool_name = tool.schema.name
        
        if tool_name in self._tools and not override:
            raise ValueError(f"Tool '{tool_name}' already registered. Use override=True to replace.")
        
        self._tools[tool_name] = tool
        
        # 按类别组织
        category = tool.schema.category
        if category not in self._categories:
            self._categories[category] = []
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)
        
        print(f"Registered tool: {tool_name} (category: {category})")
    
    def unregister(self, tool_name: str) -> None:
        """注销工具"""
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            category = tool.schema.category
            
            del self._tools[tool_name]
            
            if category in self._categories:
                self._categories[category].remove(tool_name)
                if not self._categories[category]:
                    del self._categories[category]
            
            print(f"❌ Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(tool_name)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return tool(**kwargs)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """列出所有工具"""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """列出所有类别"""
        return list(self._categories.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息"""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_info()
        return None
    
    def get_all_tools_info(self) -> Dict[str, Any]:
        """获取所有工具信息"""
        info = {
            "categories": {},
            "total_tools": len(self._tools)
        }
        
        for category, tool_names in self._categories.items():
            info["categories"][category] = []
            for tool_name in tool_names:
                tool_info = self.get_tool_info(tool_name)
                if tool_info:
                    info["categories"][category].append(tool_info)
        
        return info
    
    def search_tools(self, keyword: str) -> List[str]:
        """搜索工具（根据名称或描述）"""
        results = []
        keyword_lower = keyword.lower()
        
        for tool_name, tool in self._tools.items():
            if (keyword_lower in tool_name.lower() or 
                keyword_lower in tool.schema.description.lower()):
                results.append(tool_name)
        
        return results


# 全局工具注册实例
tool_registry = ToolRegistry()


def register_tool(tool_class: type) -> type:
    """装饰器：自动注册工具类"""
    if issubclass(tool_class, BaseTool):
        tool_instance = tool_class()
        tool_registry.register(tool_instance)
    return tool_class
