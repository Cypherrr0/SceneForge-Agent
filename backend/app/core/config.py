"""
核心配置模块
"""
import os
import json
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置"""
    
    # API配置
    APP_NAME: str = "Hunyuan3D Agent API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # 服务器配置 - 用于生成完整的资源URL
    SERVER_HOST: str = "0.0.0.0"  # 监听地址
    SERVER_PORT: int = 8080
    # 外部访问地址，用于生成资源URL（使用空字符串表示相对路径，通过Next.js rewrites转发）
    PUBLIC_URL: str = ""  # 使用相对路径，不依赖具体IP地址
    
    # CORS配置
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://10.21.5.201:3000",  # 服务器IP
        "http://10.21.5.201:5173",  # 服务器IP (备用端口)
    ]
    
    # API密钥 - 从config.json读取
    QWEN_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置绝对路径
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        project_root = os.path.abspath(os.path.join(backend_dir, ".."))
        self.OUTPUT_DIR = os.path.join(project_root, "outputs")
        self.SESSION_DIR = os.path.join(project_root, "sessions")
        # 从项目根目录的config/config.json读取API密钥
        self._load_api_keys_from_config()
    
    # LLM模型配置
    DECISION_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"
    SCRIPT_GENERATION_MODEL: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    
    # 3D生成配置
    MAX_ITERATIONS: int = 3
    SCORE_THRESHOLD: int = 60
    
    # 文件存储配置（使用绝对路径）
    OUTPUT_DIR: str = ""
    SESSION_DIR: str = ""
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # 会话配置
    SESSION_TIMEOUT: int = 3600  # 1小时
    MAX_SESSIONS: int = 100
    
    # 任务执行配置
    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT: int = 1800  # 30分钟
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL: int = 30  # 秒
    
    def _load_api_keys_from_config(self):
        """从config/config.json加载API密钥"""
        # 从backend目录向上找到项目根目录的config/config.json
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_file = os.path.join(backend_dir, "..", "config", "config.json")
        config_file = os.path.abspath(config_file)
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.QWEN_API_KEY = config.get('qwen_api_key', '')
                    self.GEMINI_API_KEY = config.get('gemini_api_key', '')
                    print(f"从配置文件加载API密钥: {config_file}")
                    if not self.QWEN_API_KEY or not self.GEMINI_API_KEY:
                        print("警告: API密钥未配置或为空")
            except Exception as e:
                print(f"读取配置文件失败: {e}")
        else:
            print(f"配置文件不存在: {config_file}")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()

