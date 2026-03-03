"""
会话管理服务
"""
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path

from app.models import Session, SessionStatus
from app.core import settings


class SessionService:
    """会话管理服务"""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._session_dir = Path(settings.SESSION_DIR)
        self._session_dir.mkdir(parents=True, exist_ok=True)
        
        # 启动时加载持久化的会话
        self._load_sessions()
    
    def create_session(self, user_prompt: str, interactive: bool = False) -> Session:
        """
        创建新会话
        
        Args:
            user_prompt: 用户输入提示词
            interactive: 是否交互模式
            
        Returns:
            Session: 新创建的会话
        """
        # 生成会话ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        session_id = f"session_{timestamp}_{unique_id}"
        
        # 创建输出目录
        output_dir = os.path.join(settings.OUTPUT_DIR, session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建会话对象
        session = Session(
            session_id=session_id,
            user_prompt=user_prompt,
            status=SessionStatus.IDLE,
            interactive=interactive,
            output_dir=output_dir,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 存储会话
        self._sessions[session_id] = session
        self._save_session(session)
        
        # 清理过期会话
        self._cleanup_expired_sessions()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def update_session(self, session: Session) -> None:
        """更新会话"""
        session.updated_at = datetime.now()
        self._sessions[session.session_id] = session
        self._save_session(session)
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            
            # 删除持久化文件
            session_file = self._session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            return True
        return False
    
    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Session]:
        """列出会话"""
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True
        )
        return sessions[offset:offset + limit]
    
    def get_session_count(self) -> int:
        """获取会话总数"""
        return len(self._sessions)
    
    def _save_session(self, session: Session) -> None:
        """持久化会话到磁盘"""
        session_file = self._session_dir / f"{session.session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"保存会话失败 {session.session_id}: {e}")
    
    def _load_sessions(self) -> None:
        """从磁盘加载会话"""
        if not self._session_dir.exists():
            return
        
        for session_file in self._session_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session = Session(**data)
                    self._sessions[session.session_id] = session
            except Exception as e:
                print(f"加载会话失败 {session_file}: {e}")
    
    def _cleanup_expired_sessions(self) -> None:
        """清理过期会话"""
        if len(self._sessions) <= settings.MAX_SESSIONS:
            return
        
        # 按时间排序，删除最旧的会话
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.created_at
        )
        
        to_delete = len(sessions) - settings.MAX_SESSIONS
        for session in sessions[:to_delete]:
            self.delete_session(session.session_id)
        
        # 删除超时的会话
        timeout_threshold = datetime.now() - timedelta(seconds=settings.SESSION_TIMEOUT)
        expired_sessions = [
            s for s in self._sessions.values()
            if s.updated_at < timeout_threshold and s.status not in [SessionStatus.EXECUTING]
        ]
        
        for session in expired_sessions:
            self.delete_session(session.session_id)


# 全局单例
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """获取会话服务单例"""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service

