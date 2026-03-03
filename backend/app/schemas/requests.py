"""
API请求Schema
"""
from typing import Optional
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    prompt: str = Field(..., description="用户输入的生成提示词", min_length=1, max_length=2000)
    interactive: bool = Field(default=False, description="是否启用交互模式")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "一只可爱的猫咪坐在椅子上",
                "interactive": False
            }
        }


class StartGenerationRequest(BaseModel):
    """开始生成请求"""
    session_id: str = Field(..., description="会话ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_20231120_123456_abc123"
            }
        }


class ContinueExecutionRequest(BaseModel):
    """继续执行请求（交互模式）"""
    session_id: str = Field(..., description="会话ID")
    user_feedback: Optional[str] = Field(default=None, description="用户反馈或修改建议")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_20231120_123456_abc123",
                "user_feedback": "继续执行下一步"
            }
        }


class CancelSessionRequest(BaseModel):
    """取消会话请求"""
    session_id: str = Field(..., description="会话ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_20231120_123456_abc123"
            }
        }


class UserInteractionRequest(BaseModel):
    """用户交互响应请求"""
    session_id: str = Field(..., description="会话ID")
    action: str = Field(..., description="用户选择的操作: 'continue', 'stop', 'replan'")
    message: Optional[str] = Field(default=None, description="用户输入的消息（重新规划时使用）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_20231120_123456_abc123",
                "action": "continue",
                "message": None
            }
        }

