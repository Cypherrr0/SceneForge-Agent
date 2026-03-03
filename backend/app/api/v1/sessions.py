"""
会话管理API路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from app.schemas import (
    CreateSessionRequest,
    StartGenerationRequest,
    ContinueExecutionRequest,
    CancelSessionRequest,
    UserInteractionRequest,
    SessionResponse,
    SessionListResponse,
    GenerationStatusResponse,
    GenerationResultResponse,
    BaseResponse
)
from app.models import SessionStatus
from app.services import get_session_service, get_task_executor
from app.services.websocket_manager import get_websocket_manager

router = APIRouter()


@router.post("/cancel-all", response_model=BaseResponse)
async def cancel_all_sessions():
    """取消所有正在运行的会话"""
    try:
        task_executor = get_task_executor()
        session_service = get_session_service()
        ws_manager = get_websocket_manager()
        
        # 取消所有任务
        count = task_executor.cancel_all_tasks()
        
        # 获取所有正在执行的会话并更新状态
        running_sessions = []
        for session_id in task_executor.get_running_tasks():
            session = session_service.get_session(session_id)
            if session and session.status in [SessionStatus.PARSING, SessionStatus.PLANNING, SessionStatus.EXECUTING]:
                session.status = SessionStatus.CANCELLED
                session.error_message = "任务已被用户批量取消"
                session_service.update_session(session)
                running_sessions.append(session_id)
                
                # 通知客户端
                await ws_manager.send_to_session(session_id, {
                    "type": "cancelled",
                    "message": "任务已被批量取消"
                })
        
        return BaseResponse(
            success=True,
            message=f"已取消 {count} 个正在运行的任务"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量取消失败: {str(e)}")


@router.post("/create", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """创建新会话"""
    try:
        session_service = get_session_service()
        session = session_service.create_session(
            user_prompt=request.prompt,
            interactive=request.interactive
        )
        
        return SessionResponse(
            success=True,
            message="会话创建成功",
            session=session
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@router.post("/start", response_model=BaseResponse)
async def start_generation(request: StartGenerationRequest, background_tasks: BackgroundTasks):
    """开始生成任务"""
    try:
        session_service = get_session_service()
        session = session_service.get_session(request.session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        if session.status not in [SessionStatus.IDLE, SessionStatus.FAILED]:
            raise HTTPException(status_code=400, detail=f"会话状态无效: {session.status}")
        
        # 更新状态
        session.status = SessionStatus.EXECUTING
        session_service.update_session(session)
        
        # 在后台执行任务
        background_tasks.add_task(execute_generation_task, request.session_id)
        
        return BaseResponse(
            success=True,
            message="生成任务已启动"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")


@router.get("/status/{session_id}", response_model=GenerationStatusResponse)
async def get_generation_status(session_id: str):
    """获取生成状态"""
    try:
        session_service = get_session_service()
        session = session_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        completed_steps = sum(1 for step in session.steps if step.status.value == "completed")
        
        return GenerationStatusResponse(
            success=True,
            session_id=session.session_id,
            status=session.status,
            current_step=session.steps[session.current_step_index].name if session.current_step_index < len(session.steps) else None,
            progress=session.overall_progress,
            total_steps=session.total_steps,
            completed_steps=completed_steps
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/result/{session_id}", response_model=GenerationResultResponse)
async def get_generation_result(session_id: str):
    """获取生成结果"""
    try:
        session_service = get_session_service()
        session = session_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        if session.status not in [SessionStatus.COMPLETED, SessionStatus.FAILED]:
            raise HTTPException(status_code=400, detail="任务尚未完成")
        
        execution_time = None
        if session.completed_at and session.started_at:
            execution_time = (session.completed_at - session.started_at).total_seconds()
        
        return GenerationResultResponse(
            success=session.status == SessionStatus.COMPLETED,
            session_id=session.session_id,
            status=session.status,
            result=session.result.model_dump(),
            execution_time=execution_time
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")


@router.get("/list", response_model=SessionListResponse)
async def list_sessions(limit: int = 50, offset: int = 0):
    """列出会话"""
    try:
        session_service = get_session_service()
        sessions = session_service.list_sessions(limit=limit, offset=offset)
        total = session_service.get_session_count()
        
        sessions_data = [
            {
                "session_id": s.session_id,
                "user_prompt": s.user_prompt,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "progress": s.overall_progress
            }
            for s in sessions
        ]
        
        return SessionListResponse(
            success=True,
            sessions=sessions_data,
            total=total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出会话失败: {str(e)}")


@router.post("/cancel", response_model=BaseResponse)
async def cancel_session(request: CancelSessionRequest):
    """取消会话"""
    try:
        session_service = get_session_service()
        task_executor = get_task_executor()
        ws_manager = get_websocket_manager()
        
        session = session_service.get_session(request.session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        if session.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED]:
            return BaseResponse(
                success=True,
                message="会话已结束"
            )
        
        # 取消正在执行的任务
        cancelled = task_executor.cancel_task(request.session_id)
        
        # 更新会话状态
        session.status = SessionStatus.CANCELLED
        session.error_message = "任务已被用户取消"
        session_service.update_session(session)
        
        # 通知WebSocket客户端
        await ws_manager.send_to_session(session.session_id, {
            "type": "cancelled",
            "message": "任务已取消"
        })
        
        return BaseResponse(
            success=True,
            message=f"会话已取消 (任务执行器{'已标记取消' if cancelled else '未在运行'})"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消会话失败: {str(e)}")


@router.post("/interact", response_model=BaseResponse)
async def user_interaction(request: UserInteractionRequest):
    """
    处理用户交互响应（交互模式）
    
    action 可以是:
    - 'continue' 或 'y' 或 '': 继续下一步
    - 'stop' 或 'n': 停止执行
    - 'replan' 或 'r': 重新规划
    """
    try:
        task_executor = get_task_executor()
        session_service = get_session_service()
        
        session = session_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 映射前端的action到后端expect的响应格式
        action_map = {
            'continue': '',  # 空字符串或y表示继续
            'y': '',
            'stop': 'n',
            'n': 'n',
            'replan': 'r',
            'r': 'r'
        }
        
        response_value = action_map.get(request.action.lower(), '')
        
        # 发送响应到等待的任务
        success = task_executor.send_user_response(request.session_id, response_value)
        
        if not success:
            return BaseResponse(
                success=False,
                message="会话未在等待用户响应"
            )
        
        # 如果是重新规划且有message，需要连续发送两次：
        # 第一次'r'已经发送（上面），现在发送第二次（用户的具体需求）
        if request.action.lower() in ['replan', 'r'] and request.message:
            import asyncio
            import time
            
            # 等待第一个input()消费完'r'并进入第二个input()
            # 使用短暂延迟确保顺序正确
            await asyncio.sleep(0.1)
            
            # 发送用户的具体需求到第二个input()
            success2 = task_executor.send_user_response(request.session_id, request.message)
            if not success2:
                print(f"⚠️ [Sessions] 发送重新规划需求失败，但第一次响应已发送")
        
        return BaseResponse(
            success=True,
            message=f"已发送用户响应: {request.action}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理用户交互失败: {str(e)}")


async def execute_generation_task(session_id: str):
    """执行生成任务（后台任务）"""
    try:
        session_service = get_session_service()
        task_executor = get_task_executor()
        ws_manager = get_websocket_manager()
        
        # 获取会话
        session = session_service.get_session(session_id)
        if not session:
            return
        
        # 注册回调
        async def on_status_update(data: dict):
            await ws_manager.send_to_session(session_id, {
                "type": "status_update",
                **data
            })
            
            # 更新会话
            session = session_service.get_session(session_id)
            if session and "status" in data:
                session.status = data["status"]
                session_service.update_session(session)
        
        async def on_llm_message(data: dict):
            """处理LLM对话消息"""
            await ws_manager.send_llm_message(
                session_id, 
                data.get("content") or data.get("message", "")
            )
        
        async def on_user_input_required(data: dict):
            """处理需要用户输入的请求"""
            # 获取最新的session状态
            current_session = session_service.get_session(session_id)
            # 发送交互请求，包含完整的session信息
            await ws_manager.send_user_input_required(
                session_id,
                data.get("prompt", "请选择操作"),
                current_session.model_dump(mode='json') if current_session and hasattr(current_session, 'model_dump') else None
            )
        
        async def on_log_message(data: dict):
            """处理技术性日志消息"""
            await ws_manager.send_to_session(session_id, {
                "type": "log_message",
                "message": data.get("content") or data.get("message", ""),
                "content": data.get("content") or data.get("message", ""),
                "timestamp": datetime.now().isoformat()
            })
        
        task_executor.register_callback("status_update", on_status_update)
        task_executor.register_callback("llm_message", on_llm_message)
        task_executor.register_callback("user_input_required", on_user_input_required)
        task_executor.register_callback("log_message", on_log_message)
        
        # 执行任务
        updated_session = await task_executor.execute_generation(session)
        
        # 保存结果
        session_service.update_session(updated_session)
        
        # 通知完成
        if updated_session.status == SessionStatus.COMPLETED:
            session_dict = updated_session.model_dump(mode='json') if hasattr(updated_session, 'model_dump') else updated_session.dict()
            await ws_manager.send_completion(
                session_id, 
                updated_session.result.model_dump() if hasattr(updated_session.result, 'model_dump') else updated_session.result.dict(),
                session_dict
            )
        else:
            await ws_manager.send_error(session_id, updated_session.error_message or "未知错误")
        
    except Exception as e:
        print(f"后台任务执行失败: {e}")
        import traceback
        traceback.print_exc()

