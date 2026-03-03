"""
会话数据模型
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """会话状态"""
    IDLE = "idle"
    PARSING = "parsing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStep(BaseModel):
    """任务步骤"""
    step_id: str
    name: str
    description: str
    tool: str
    status: StepStatus = StepStatus.PENDING
    progress: float = 0.0  # 0-100
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GenerationIntent(BaseModel):
    """生成意图"""
    generation_prompt: str
    wants: Dict[str, bool]
    constraints: Dict[str, bool]
    rendering_complexity: Dict[str, Any]
    reason: str


class GenerationResult(BaseModel):
    """生成结果"""
    optimized_prompt: Optional[str] = None
    image_2d_url: Optional[str] = None
    model_3d_url: Optional[str] = None
    render_views: List[str] = Field(default_factory=list)
    blend_file_url: Optional[str] = None
    video_url: Optional[str] = None
    quality_score: Optional[float] = None
    evaluation_details: Optional[Dict[str, Any]] = None
    
    class Config:
        protected_namespaces = ()  # 允许model_开头的字段名


class Session(BaseModel):
    """生成会话"""
    session_id: str
    user_prompt: str
    status: SessionStatus = SessionStatus.IDLE
    
    # 意图和计划
    intent: Optional[GenerationIntent] = None
    plan: Optional[Dict[str, Any]] = None
    steps: List[TaskStep] = Field(default_factory=list)
    
    # 执行进度
    current_step_index: int = 0
    total_steps: int = 0
    overall_progress: float = 0.0
    
    # 结果
    result: GenerationResult = Field(default_factory=GenerationResult)
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 错误信息
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # 交互模式
    interactive: bool = False
    waiting_for_user: bool = False
    
    # 输出目录
    output_dir: str = ""

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

