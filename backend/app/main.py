"""
FastAPI主应用
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core import settings
from app.api.v1 import sessions, websocket
from app.schemas import HealthCheckResponse
from app.services import get_task_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("🚀 启动Hunyuan3D Agent后端服务...")
    
    # 确保输出目录存在
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.SESSION_DIR, exist_ok=True)
    
    # 预热任务执行器（延迟初始化Agent）
    print("⏳ 初始化任务执行器...")
    # task_executor = get_task_executor()  # 延迟到第一次使用时初始化
    
    print("✅ 服务启动完成")
    
    yield
    
    # 关闭时
    print("🛑 关闭服务...")
    executor = get_task_executor()
    executor.shutdown()
    print("✅ 服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="智能3D生成后端API",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（输出文件）
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# API路由
app.include_router(
    sessions.router,
    prefix=f"{settings.API_V1_PREFIX}/sessions",
    tags=["Sessions"]
)

# WebSocket路由
app.include_router(
    websocket.router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["WebSocket"]
)


@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "Hunyuan3D Agent API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        services={
            "api": "running",
            "websocket": "running",
            "storage": "available"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # 监听所有IP
        port=8080,
        reload=False,  # 生产环境关闭自动重载
        log_level="info"
    )

