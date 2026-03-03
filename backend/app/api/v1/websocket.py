"""
WebSocket API路由
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import get_websocket_manager, WebSocketManager
from app.services.task_executor import get_task_executor
from app.services.session_service import get_session_service
from app.models import SessionStatus

router = APIRouter()


def _handle_disconnect(ws_manager: WebSocketManager, websocket: WebSocket, session_id: str, reason: str):
    """
    处理 WebSocket 断开连接的统一逻辑
    
    Args:
        ws_manager: WebSocket 管理器
        websocket: WebSocket 连接对象
        session_id: 会话ID
        reason: 断开原因（用于日志）
    """
    print(f"📡 [WebSocket] 检测到连接断开 ({reason}): session={session_id}")
    
    # 获取任务执行器和会话服务
    task_executor = get_task_executor()
    session_service = get_session_service()
    
    # 先检查任务是否在运行（包括取消中的任务）
    is_task_running = session_id in task_executor.get_running_tasks(include_cancelling=True)
    running_tasks = task_executor.get_running_tasks(include_cancelling=False)
    cancelling_tasks = task_executor.get_cancelling_tasks()
    
    print(f"📊 [WebSocket] 断开前状态:")
    print(f"   - 该任务是否存活: {'✅ 是' if is_task_running else '❌ 否'}")
    print(f"   - 运行中的任务: {running_tasks}")
    print(f"   - 取消中的任务: {cancelling_tasks}")
    
    # 断开连接，如果返回了 session_id 说明该 session 没有其他连接了
    disconnected_session_id = ws_manager.disconnect(websocket)
    
    print(f"📡 [WebSocket] disconnect() 返回值: {disconnected_session_id}")
    
    if disconnected_session_id:
        print(f"🔌 [WebSocket] 该 session 所有连接已断开，准备取消任务: {disconnected_session_id}")
        
        # 尝试取消正在运行的任务
        cancelled = task_executor.cancel_task(disconnected_session_id)
        
        if cancelled:
            print(f"✅ [WebSocket] 已自动取消任务: {disconnected_session_id}")
            
            # 更新会话状态
            session = session_service.get_session(disconnected_session_id)
            if session and session.status in [SessionStatus.PARSING, SessionStatus.PLANNING, SessionStatus.EXECUTING]:
                session.status = SessionStatus.CANCELLED
                session.error_message = f"连接断开（{reason}），任务已自动取消"
                session_service.update_session(session)
                print(f"📝 [WebSocket] 已更新会话状态为已取消: {disconnected_session_id}")
            
            # 打印剩余的存活任务（区分运行中和取消中）
            running_tasks = task_executor.get_running_tasks(include_cancelling=False)
            cancelling_tasks = task_executor.get_cancelling_tasks()
            print(f"📋 [WebSocket] 取消后的任务状态:")
            print(f"   - 仍在运行: {running_tasks}")
            print(f"   - 正在取消: {cancelling_tasks}")
        else:
            print(f"ℹ️ [WebSocket] 任务未在运行中（可能已完成、已失败或从未启动）: {disconnected_session_id}")
            print(f"📋 [WebSocket] 当前存活的任务列表: {task_executor.get_running_tasks()}")
    else:
        print(f"ℹ️ [WebSocket] 该 session 还有其他活跃连接，不取消任务: {session_id}")
        print(f"📋 [WebSocket] 当前存活的任务列表: {task_executor.get_running_tasks()}")
    
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket连接端点"""
    ws_manager = get_websocket_manager()
    
    await ws_manager.connect(websocket, session_id)
    
    try:
        while True:
            # 接收客户端消息（心跳）
            data = await websocket.receive_text()
            
            # 回复心跳
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        _handle_disconnect(ws_manager, websocket, session_id, "正常断开")
    
    except Exception as e:
        print(f"❌ [WebSocket] WebSocket错误: {e}")
        import traceback
        traceback.print_exc()
        
        _handle_disconnect(ws_manager, websocket, session_id, f"异常断开: {type(e).__name__}")

