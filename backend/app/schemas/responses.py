"""
API响应Schema
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from app.models import Session, SessionStatus, StepStatus


class BaseResponse(BaseModel):
    """基础响应"""
    success: bool
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseResponse):
    """错误响应"""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class SessionResponse(BaseResponse):
    """会话响应"""
    session: Session


class SessionListResponse(BaseResponse):
    """会话列表响应"""
    sessions: List[Dict[str, Any]]
    total: int


class GenerationStatusResponse(BaseResponse):
    """生成状态响应"""
    session_id: str
    status: SessionStatus
    current_step: Optional[str] = None
    progress: float
    total_steps: int
    completed_steps: int
    estimated_time_remaining: Optional[int] = None  # 秒


class GenerationResultResponse(BaseResponse):
    """生成结果响应"""
    session_id: str
    status: SessionStatus
    result: Dict[str, Any]
    execution_time: Optional[float] = None  # 秒


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

