"""
WebSocket连接管理器
"""
import asyncio
import json
from typing import Dict, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # session_id -> Set[WebSocket]
        self._connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> session_id
        self._reverse_map: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """接受WebSocket连接"""
        await websocket.accept()
        
        if session_id not in self._connections:
            self._connections[session_id] = set()
        
        self._connections[session_id].add(websocket)
        self._reverse_map[websocket] = session_id
        
        print(f"✅ WebSocket连接建立: session={session_id}, total={len(self._connections[session_id])}")
        
        # 发送欢迎消息
        await self.send_to_session(session_id, {
            "type": "connected",
            "message": "WebSocket连接成功",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket) -> str:
        """
        断开WebSocket连接
        
        Returns:
            str: 断开连接的 session_id，如果没有则返回 None
        """
        if websocket in self._reverse_map:
            session_id = self._reverse_map[websocket]
            
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)
                
                # 检查这个 session 是否还有其他连接
                has_other_connections = bool(self._connections[session_id])
                
                if not self._connections[session_id]:
                    del self._connections[session_id]
            else:
                has_other_connections = False
            
            del self._reverse_map[websocket]
            
            # 只有当该 session 没有其他连接时，才返回 session_id（用于取消任务）
            if not has_other_connections:
                return session_id
            else:
                return None
        
        return None
    
    async def send_to_session(self, session_id: str, message: dict):
        """发送消息到指定会话的所有连接"""
        if session_id not in self._connections:
            return
        
        # 添加时间戳
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        
        # 序列化消息
        message_text = json.dumps(message, ensure_ascii=False)
        
        # 发送到所有连接（使用副本迭代，避免并发修改）
        disconnected = set()
        for websocket in list(self._connections[session_id]):
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                print(f"发送消息失败: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """广播消息到所有连接"""
        for session_id in list(self._connections.keys()):
            await self.send_to_session(session_id, message)
    
    async def send_progress_update(self, session_id: str, progress: float, current_step: str, message: str = None):
        """发送进度更新"""
        await self.send_to_session(session_id, {
            "type": "progress",
            "progress": progress,
            "current_step": current_step,
            "message": message
        })
    
    async def send_step_complete(self, session_id: str, step_name: str, result: dict):
        """发送步骤完成通知"""
        await self.send_to_session(session_id, {
            "type": "step_complete",
            "step_name": step_name,
            "result": result
        })
    
    async def send_error(self, session_id: str, error: str, details: dict = None):
        """发送错误消息"""
        await self.send_to_session(session_id, {
            "type": "error",
            "error": error,
            "details": details or {}
        })
    
    async def send_completion(self, session_id: str, result: dict, session: dict = None):
        """发送任务完成通知"""
        message = {
            "type": "completed",
            "result": result
        }
        if session:
            message["session"] = session
        await self.send_to_session(session_id, message)
    
    async def send_llm_message(self, session_id: str, content: str):
        """发送LLM对话消息"""
        await self.send_to_session(session_id, {
            "type": "llm_message",
            "content": content,
            "message": content
        })
    
    async def send_user_input_required(self, session_id: str, prompt: str, session: dict = None):
        """发送需要用户输入的通知"""
        message = {
            "type": "user_input_required",
            "message": prompt,
            "prompt": prompt
        }
        if session:
            message["session"] = session
        await self.send_to_session(session_id, message)
    
    def get_connection_count(self, session_id: str = None) -> int:
        """获取连接数"""
        if session_id:
            return len(self._connections.get(session_id, set()))
        return sum(len(conns) for conns in self._connections.values())


# 全局单例
_ws_manager = None


def get_websocket_manager() -> WebSocketManager:
    """获取WebSocket管理器单例"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager

