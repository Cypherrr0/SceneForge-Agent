"""
任务执行器服务
负责异步执行3D生成任务
"""
import os
import sys
import asyncio
import traceback
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models import Session, SessionStatus, StepStatus, TaskStep, GenerationIntent, GenerationResult
from app.core import settings


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or settings.MAX_CONCURRENT_TASKS
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._agent = None
        self._callbacks: Dict[str, Callable] = {}
        # 记录正在执行的任务，用于取消
        self._running_tasks: Dict[str, bool] = {}  # session_id -> should_cancel
        # 用于交互模式的用户响应队列
        self._user_responses: Dict[str, asyncio.Queue] = {}  # session_id -> Queue
        # 用于存储每个会话的事件循环
        self._session_loops: Dict[str, asyncio.AbstractEventLoop] = {}  # session_id -> event_loop
    
    def _initialize_agent(self):
        """延迟初始化Agent"""
        if self._agent is not None:
            return self._agent
        
        try:
            from Hunyuan3DAgentV2 import Hunyuan3DAgentV2
            
            self._agent = Hunyuan3DAgentV2(
                qwen_api_key=settings.QWEN_API_KEY,
                gemini_api_key=settings.GEMINI_API_KEY,
                model_name=settings.DECISION_MODEL,
                script_generation_model=settings.SCRIPT_GENERATION_MODEL,
                max_iterations=settings.MAX_ITERATIONS,
                score_threshold=settings.SCORE_THRESHOLD
            )
            
            print("✅ Hunyuan3DAgentV2 初始化成功")
            return self._agent
            
        except Exception as e:
            print(f"❌ Agent初始化失败: {e}")
            traceback.print_exc()
            raise
    
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        self._callbacks[event] = callback
    
    async def _trigger_callback(self, event: str, data: Dict[str, Any]):
        """触发回调"""
        if event in self._callbacks:
            callback = self._callbacks[event]
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
    
    async def execute_generation(self, session: Session) -> Session:
        """
        执行生成任务
        
        Args:
            session: 会话对象
            
        Returns:
            Session: 更新后的会话
        """
        session_id = session.session_id
        
        try:
            # 标记任务开始执行
            self._running_tasks[session_id] = False
            print(f"🚀 [TaskExecutor] 开始执行任务: {session_id}")
            
            # 初始化交互模式的响应队列
            if session.interactive:
                self._user_responses[session_id] = asyncio.Queue()
                self._session_loops[session_id] = asyncio.get_event_loop()
            
            # 初始化Agent
            agent = self._initialize_agent()
            
            # 更新会话状态
            session.status = SessionStatus.PARSING
            session.started_at = datetime.now()
            await self._trigger_callback("status_update", {
                "session_id": session.session_id,
                "status": session.status,
                "message": "开始解析用户意图...",
                "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
            })
            
            # 构建任务描述
            task_description = self._build_task_description(session)

            # 在真正执行前固定会话级别的memory，避免重做时生成新uid/目录
            try:
                from tools.intent_parser_tools import set_session_memory
                set_session_memory(
                    uid=session.session_id,
                    output_dir=session.output_dir,
                    session_dir=session.output_dir,
                    base_save_dir=session.output_dir
                )
            except Exception:
                pass  # 静默失败，不影响主流程
            
            # 清理Agent历史（避免污染）
            agent.clear_history()
            
            # 检查是否已被取消
            if self._running_tasks.get(session_id, False):
                print(f"⛔ [TaskExecutor] 任务在启动前被取消: {session_id}")
                raise Exception("任务已被用户取消")
            
            # 使用线程池执行同步的Agent调用
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_agent_with_interaction,
                agent,
                task_description,
                session,
                session_id
            )
            
            # 检查是否在执行过程中被取消
            if self._running_tasks.get(session_id, False):
                print(f"⛔ [TaskExecutor] 任务在执行过程中被取消: {session_id}")
                raise Exception("任务已被用户取消")
            
            # 解析执行结果并发送LLM消息
            session = await self._parse_execution_result_and_notify(session, result)
            
            # 更新完成状态
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.now()
            
            print(f"✅ [TaskExecutor] 任务执行完成: {session_id}")
            
            await self._trigger_callback("status_update", {
                "session_id": session.session_id,
                "status": session.status,
                "message": "生成完成！",
                "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
            })
            
            return session
            
        except Exception as e:
            # 检查是否是取消导致的异常
            if "取消" in str(e) or "cancelled" in str(e).lower():
                print(f"🛑 [TaskExecutor] 任务已取消: {session_id}")
                session.status = SessionStatus.CANCELLED
                session.error_message = "任务已被用户取消"
            else:
                print(f"❌ [TaskExecutor] 任务执行失败: {session_id} - {e}")
                traceback.print_exc()
                session.status = SessionStatus.FAILED
                session.error_message = str(e)
                session.error_details = {
                    "traceback": traceback.format_exc()
                }
            
            await self._trigger_callback("status_update", {
                "session_id": session.session_id,
                "status": session.status,
                "error": session.error_message,
                "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
            })
            
            return session
        finally:
            # 清理任务标记和队列
            if session_id in self._running_tasks:
                del self._running_tasks[session_id]
                print(f"🧹 [TaskExecutor] 清理任务标记: {session_id}")
            if session_id in self._user_responses:
                del self._user_responses[session_id]
            if session_id in self._session_loops:
                del self._session_loops[session_id]
    
    async def wait_for_user_response(self, session_id: str, timeout: int = 3600) -> str:
        """
        等待用户的交互响应
        
        Args:
            session_id: 会话ID
            timeout: 超时时间（秒）
            
        Returns:
            str: 用户的选择 ('continue', 'stop', 'replan')
        """
        if session_id not in self._user_responses:
            return 'continue'  # 非交互模式，自动继续
        
        try:
            # 等待用户响应
            response = await asyncio.wait_for(
                self._user_responses[session_id].get(),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            print(f"⏱️ [TaskExecutor] 等待用户响应超时: {session_id}")
            return 'continue'  # 超时则自动继续
    
    def send_user_response(self, session_id: str, response: str) -> bool:
        """
        发送用户响应到等待的任务
        
        Args:
            session_id: 会话ID
            response: 用户的选择
            
        Returns:
            bool: 是否成功发送
        """
        if session_id not in self._user_responses:
            print(f"❌ [TaskExecutor] 会话不在等待用户响应: {session_id}")
            return False
        
        if session_id not in self._session_loops:
            print(f"❌ [TaskExecutor] 找不到会话的事件循环: {session_id}")
            return False
        
        # 在会话的事件循环中放入响应
        loop = self._session_loops[session_id]
        asyncio.run_coroutine_threadsafe(
            self._user_responses[session_id].put(response),
            loop
        )
        print(f"✅ [TaskExecutor] 已发送用户响应: {session_id} -> {response}")
        return True
    
    def _execute_agent_with_interaction(self, agent, task_description: str, session: Session, session_id: str) -> Dict[str, Any]:
        """同步执行Agent（在线程池中运行，支持Web交互）"""
        import builtins
        import sys
        from io import StringIO
        
        # 保存原始的 input 函数和 stdout
        original_input = builtins.input
        original_stdout = sys.stdout
        
        # 创建一个缓冲区来捕获输出
        captured_output = StringIO()
        
        # 保存 self 的引用，以便在嵌套类中使用
        task_executor = self
        
        # 创建一个自定义的输出流，同时写入原始 stdout 和捕获缓冲区
        class TeeOutput:
            def __init__(self, *streams):
                self.streams = streams
            
            def write(self, data):
                # 先写入所有 streams（确保输出不丢失）
                for stream in self.streams:
                    try:
                        stream.write(data)
                        stream.flush()
                    except (OSError, ValueError) as e:
                        # 忽略文件描述符错误，避免中断执行
                        pass
                
                # 将输出发送到前端
                if data.strip() and session_id in task_executor._session_loops:
                    loop = task_executor._session_loops[session_id]
                    
                    # 排除交互提示和分隔线（完全不发送）
                    noise_patterns = [
                        "请选择操作:", "[回车/y]", "[n]", "[r]", "> ",
                        "---", "===", "━━━"
                    ]
                    
                    # 只是不发送到前端，但不影响本地输出
                    if any(x in data for x in noise_patterns):
                        return
                    
                    # 技术性日志模式（发送到实时日志，不显示在聊天框）
                    log_patterns = [
                        "Calling tool:", "执行成功", "执行失败",
                        "交互模式：", "本轮已执行工具", "等待用户确认",
                        "本轮工具调用:", "parse_user_intent:", "create_task_plan:",
                        "Success 成功", "准备工作完成:", "✅ 收到用户响应:",
                        "已完成工具:", "执行轮次", "继续执行下一步",
                        "还未创建任务计划", "准备构建下一轮输入", "当前状态:",
                        "检测到需要继续执行", "进入下一轮"
                    ]
                    
                    # 判断是技术性日志还是用户友好消息
                    is_log = any(x in data for x in log_patterns)
                    message_type = "log_message" if is_log else "llm_message"
                    
                    try:
                        asyncio.run_coroutine_threadsafe(
                            task_executor._trigger_callback(message_type, {
                                "session_id": session_id,
                                "type": message_type,
                                "content": data.strip(),
                                "message": data.strip()
                            }),
                            loop
                        )
                        
                        # 实时检测并提取资源路径
                        resource_indicators = [
                            "图像路径:", "image_path", 
                            "3D模型:", "glb_path", "mesh_path", "final_save_path",  # 添加 final_save_path
                            "blend_file", "Blend文件", "script_path"
                        ]
                        if any(indicator in data for indicator in resource_indicators):
                            asyncio.run_coroutine_threadsafe(
                                task_executor._extract_and_update_result_from_output(session_id, data, session),
                                loop
                            )
                        
                        # 实时追踪task_plan的进度和状态
                        asyncio.run_coroutine_threadsafe(
                            task_executor._update_progress_from_output(session_id, data, session),
                            loop
                        )
                    except Exception as e:
                        # 静默处理异常，避免影响主流程
                        pass
            
            def flush(self):
                for stream in self.streams:
                    try:
                        stream.flush()
                    except (OSError, ValueError) as e:
                        # 忽略文件描述符错误
                        pass
        
        # 创建自定义的 input 函数用于Web交互
        def web_input(prompt=""):
            """替代 input() 的Web交互函数"""
            print(prompt)  # 输出提示到日志
            
            if not session.interactive or session_id not in self._user_responses:
                # 非交互模式，自动继续
                return ""
            
            # 发送交互请求到前端
            loop = self._session_loops[session_id]
            
            # 触发回调，通知前端需要用户输入（包含完整的session对象，让前端能显示最新资源）
            asyncio.run_coroutine_threadsafe(
                self._trigger_callback("user_input_required", {
                    "session_id": session_id,
                    "prompt": prompt,
                    "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
                }),
                loop
            )
            
            # 等待用户响应
            try:
                # 使用同步方式等待（因为在线程池中）
                future = asyncio.run_coroutine_threadsafe(
                    self._user_responses[session_id].get(),
                    loop
                )
                response = future.result(timeout=3600)  # 1小时超时
                print(f"✅ 收到用户响应: {response}")
                return response
            except Exception as e:
                print(f"❌ 等待用户响应失败: {e}")
                return ""  # 失败则自动继续
        
        try:
            # 替换 input 函数和 stdout
            builtins.input = web_input
            sys.stdout = TeeOutput(original_stdout, captured_output)
            
            # 执行 Agent（启用交互模式）
            result = agent.decision_engine.decide_and_execute_continuous(
                user_input=task_description,
                max_rounds=30,
                interactive=session.interactive
            )
            return result
        except Exception as e:
            print(f"❌ Agent执行异常: {e}")
            traceback.print_exc()
            raise
        finally:
            # 恢复原始的 input 函数和 stdout
            builtins.input = original_input
            sys.stdout = original_stdout
    
    async def _extract_and_update_result_from_output(self, session_id: str, output: str, session: Session):
        """从输出中实时提取资源路径并更新session"""
        try:
            import re
            updated = False
            
            # 提取图像路径 - 多种模式
            image_patterns = [
                r'图像路径:\s*([^\s\n]+)',
                r'image_path["\']?\s*[:=]\s*["\']?([^"\'}\s,\n]+)',
            ]
            for pattern in image_patterns:
                match = re.search(pattern, output)
                if match:
                    image_path = match.group(1).strip('\'"')
                    if os.path.exists(image_path):
                        rel_path = os.path.relpath(image_path, settings.OUTPUT_DIR)
                        new_url = f"/outputs/{rel_path}"
                        if session.result.image_2d_url != new_url:
                            session.result.image_2d_url = new_url
                            updated = True
                            print(f"✅ 实时更新图像URL: {session.result.image_2d_url}")
                        break
            
            # 提取GLB路径 - 多种模式
            glb_patterns = [
                r'3D模型:\s*([^\s\n]+)',
                r'final_save_path:\s*([^\s\n]+)',  # 匹配 final_save_path: 输出
                r'glb_path["\']?\s*[:=]\s*["\']?([^"\'}\s,\n]+)',
                r'mesh_path["\']?\s*[:=]\s*["\']?([^"\'}\s,\n]+)',
            ]
            for pattern in glb_patterns:
                match = re.search(pattern, output)
                if match:
                    glb_path = match.group(1).strip('\'"')
                    # 只处理.glb文件
                    if glb_path.endswith('.glb') and os.path.exists(glb_path):
                        rel_path = os.path.relpath(glb_path, settings.OUTPUT_DIR)
                        new_url = f"/outputs/{rel_path}"
                        if session.result.model_3d_url != new_url:
                            session.result.model_3d_url = new_url
                            updated = True
                            print(f"✅ 实时更新3D模型URL: {session.result.model_3d_url}")
                        break
            
            # 提取Blend文件路径 - 多种模式
            blend_patterns = [
                r'Blend文件:\s*([^\s\n]+)',
                r'blend_file["\']?\s*[:=]\s*["\']?([^"\'}\s,\n]+)',
                r'blend_path["\']?\s*[:=]\s*["\']?([^"\'}\s,\n]+)',
                r'script_path["\']?\s*[:=]\s*["\']?([^"\'}\s,\n]+\.blend)',
            ]
            for pattern in blend_patterns:
                match = re.search(pattern, output)
                if match:
                    blend_path = match.group(1).strip('\'"')
                    if blend_path.endswith('.blend') and os.path.exists(blend_path):
                        rel_path = os.path.relpath(blend_path, settings.OUTPUT_DIR)
                        new_url = f"/outputs/{rel_path}"
                        if session.result.blend_file_url != new_url:
                            session.result.blend_file_url = new_url
                            updated = True
                            print(f"✅ 实时更新Blend文件URL: {session.result.blend_file_url}")
                        break
            
            # 如果有更新，发送status_update
            if updated:
                await self._trigger_callback("status_update", {
                    "session_id": session_id,
                    "status": session.status,
                    "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
                })
        except Exception as e:
            print(f"实时提取资源失败: {e}")
    
    async def _update_progress_from_output(self, session_id: str, output: str, session: Session):
        """从输出中实时追踪task_plan的进度和状态"""
        try:
            import re
            
            # 获取当前task_plan
            from tools.intent_parser_tools import get_current_plan, get_completed_steps
            current_plan = get_current_plan()
            
            # 检测task_plan创建
            if "检测到任务计划:" in output or "create_task_plan: Success" in output:
                match = re.search(r'共(\d+)个步骤', output)
                if match and current_plan:
                    total_steps = int(match.group(1))
                    session.total_steps = total_steps
                    session.status = SessionStatus.PLANNING
                    session.overall_progress = 0.0
                    
                    await self._trigger_callback("status_update", {
                        "session_id": session_id,
                        "status": session.status,
                        "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
                    })
                    # print(f"📋 已识别任务计划: {total_steps}个步骤")
            
            # 检测工具执行完成 - 匹配各种工具名称
            tool_patterns = [
                r'(optimize_3d_prompt|OptimizePromptTool).*?Success',
                r'(text_to_image|TextToImageTool).*?Success',
                r'(img_to_3d_complete|ImageTo3DCompleteTool).*?Success',
                r'(render_3d_scene|RenderSceneTool).*?Success',
                r'(evaluate_3d_asset|EvaluateAssetTool).*?Success',
            ]
            
            for pattern in tool_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    if current_plan and session.total_steps > 0:
                        # 获取已完成的步骤数
                        completed_steps = get_completed_steps()
                        # 过滤掉准备工具（parse_user_intent, create_task_plan）
                        work_steps = [s for s in completed_steps if s not in ['parse_user_intent', 'create_task_plan']]
                        completed_count = len(work_steps)
                        
                        # 更新状态为执行中
                        session.status = SessionStatus.EXECUTING
                        session.current_step_index = completed_count
                        
                        # 计算进度：准备阶段占10%，执行阶段占90%
                        base_progress = 10.0
                        execution_progress = (completed_count / session.total_steps) * 90.0
                        session.overall_progress = min(base_progress + execution_progress, 99.0)
                        
                        await self._trigger_callback("status_update", {
                            "session_id": session_id,
                            "status": session.status,
                            "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
                        })
                        # print(f"📊 进度更新: {completed_count}/{session.total_steps} ({session.overall_progress:.1f}%)")
                    break
            
            # 检测进度报告 - "进度: X/Y 个工作步骤"
            progress_match = re.search(r'进度:\s*(\d+)/(\d+)\s*个工作步骤', output)
            if progress_match:
                completed = int(progress_match.group(1))
                total = int(progress_match.group(2))
                
                if total > 0:
                    session.total_steps = total
                    session.current_step_index = completed
                    session.status = SessionStatus.EXECUTING
                    
                    # 计算进度
                    base_progress = 10.0
                    execution_progress = (completed / total) * 90.0
                    session.overall_progress = min(base_progress + execution_progress, 99.0)
                    
                    await self._trigger_callback("status_update", {
                        "session_id": session_id,
                        "status": session.status,
                        "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
                    })
                    # print(f"📊 进度更新: {completed}/{total} ({session.overall_progress:.1f}%)")
        
        except Exception as e:
            # 静默失败，不影响主流程
            pass
    
    def _build_task_description(self, session: Session) -> str:
        """构建任务描述"""
        task_description = f"""
生成3D模型：{session.user_prompt}

Session信息：
- UID: {session.session_id}
- 输出目录: {session.output_dir}

要求：
1. 先调用parse_user_intent解析用户意图
2. 然后调用create_task_plan制定计划
3. 按计划执行所有步骤：
   - 优化提示词
   - 生成2D图像（保存到 {session.output_dir}）
   - 转换为3D模型（使用参数: uid="{session.session_id}", save_dir="{session.output_dir}"）
   - 渲染多视角图像（保存到 {session.output_dir}）

重要说明：
- 所有文件统一保存到输出目录：{session.output_dir}
- 调用 img_to_3d_complete 时，save_dir 参数必须设置为 "{session.output_dir}"
- 调用渲染工具时，script_output_dir 参数设置为 "{session.output_dir}"

立即开始执行。
"""
        return task_description
    
    async def _parse_execution_result_and_notify(self, session: Session, result: Dict[str, Any]) -> Session:
        """解析执行结果并发送通知"""
        try:
            # 提取工具调用结果
            tool_calls = result.get("all_tool_calls", [])
            
            # 生成LLM对话消息映射
            tool_name_map = {
                "create_task_plan": "我已经为你制定了任务计划",
                "parse_user_intent": "我正在理解你的需求",
                "optimize_3d_prompt": "我正在优化提示词",
                "text_to_image": "我正在生成2D参考图像",
                "img_to_3d_complete": "我正在将图像转换为3D模型",
                "render_3d_scene": "我正在渲染多视角图像",
                "evaluate_3d_asset": "我正在评估3D模型质量",
            }
            
            # 更新步骤信息并发送LLM消息
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("name", "Unknown")
                content = tool_call.get("content", "")
                
                # 发送LLM消息（在工具执行前）
                llm_message = tool_name_map.get(tool_name, f"正在执行 {tool_name}")
                await self._trigger_callback("llm_message", {
                    "session_id": session.session_id,
                    "type": "llm_message",
                    "content": llm_message,
                    "message": llm_message
                })
                
                step = TaskStep(
                    step_id=f"step_{i+1}",
                    name=tool_name,
                    description=f"执行 {tool_name}",
                    tool=tool_name,
                    status=StepStatus.COMPLETED
                )
                
                # 尝试解析结果
                try:
                    if content.startswith("{") and content.endswith("}"):
                        import json
                        step.result = json.loads(content)
                        
                        # 发送工具执行完成的消息
                        if step.result.get("success"):
                            completion_msg = f"✓ {tool_name} 执行成功"
                        else:
                            completion_msg = f"✗ {tool_name} 执行失败"
                        
                        await self._trigger_callback("llm_message", {
                            "session_id": session.session_id,
                            "type": "llm_message",
                            "content": completion_msg,
                            "message": completion_msg
                        })
                except:
                    step.result = {"raw_content": content}
                
                session.steps.append(step)
                
                # 更新进度
                session.current_step_index = i + 1
                session.overall_progress = (i + 1) / len(tool_calls) * 100
                
                # 实时提取当前已完成工具的生成结果
                session.result = self._extract_generation_result(tool_calls[:i+1], session.output_dir)
                
                await self._trigger_callback("status_update", {
                    "session_id": session.session_id,
                    "status": session.status,
                    "session": session.model_dump(mode='json') if hasattr(session, 'model_dump') else session.dict()
                })
            
            # 最终再次提取完整的生成结果
            session.result = self._extract_generation_result(tool_calls, session.output_dir)
            
            # 更新最终进度
            session.total_steps = len(tool_calls)
            session.overall_progress = 100.0
            
        except Exception as e:
            print(f"解析执行结果失败: {e}")
            traceback.print_exc()
        
        return session
    
    def _parse_execution_result(self, session: Session, result: Dict[str, Any]) -> Session:
        """解析执行结果（同步版本，用于兼容）"""
        try:
            # 提取工具调用结果
            tool_calls = result.get("all_tool_calls", [])
            
            # 更新步骤信息
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("name", "Unknown")
                content = tool_call.get("content", "")
                
                step = TaskStep(
                    step_id=f"step_{i+1}",
                    name=tool_name,
                    description=f"执行 {tool_name}",
                    tool=tool_name,
                    status=StepStatus.COMPLETED
                )
                
                # 尝试解析结果
                try:
                    if content.startswith("{") and content.endswith("}"):
                        import json
                        step.result = json.loads(content)
                except:
                    step.result = {"raw_content": content}
                
                session.steps.append(step)
            
            # 提取生成结果
            session.result = self._extract_generation_result(tool_calls, session.output_dir)
            
            # 更新进度
            session.total_steps = len(tool_calls)
            session.current_step_index = len(tool_calls)
            session.overall_progress = 100.0
            
        except Exception as e:
            print(f"解析执行结果失败: {e}")
            traceback.print_exc()
        
        return session
    
    def _extract_generation_result(self, tool_calls: list, output_dir: str) -> GenerationResult:
        """从工具调用结果中提取生成结果"""
        result = GenerationResult()
        
        # 使用相对路径（通过Next.js rewrites转发到后端）
        base_url = ""
        
        try:
            for tool_call in tool_calls:
                content = tool_call.get("content", "")
                
                # 尝试解析JSON
                try:
                    if content.startswith("{") and content.endswith("}"):
                        import json
                        data = json.loads(content)
                        
                        # 提取优化的提示词
                        if "optimized_prompt" in data:
                            result.optimized_prompt = data["optimized_prompt"]
                        
                        # 提取图像路径
                        if "image_path" in data:
                            image_path = data["image_path"]
                            if os.path.exists(image_path):
                                rel_path = os.path.relpath(image_path, settings.OUTPUT_DIR)
                                result.image_2d_url = f"/outputs/{rel_path}"
                        
                        # 提取3D模型路径
                        if "glb_path" in data:
                            glb_path = data["glb_path"]
                            if os.path.exists(glb_path):
                                rel_path = os.path.relpath(glb_path, settings.OUTPUT_DIR)
                                result.model_3d_url = f"/outputs/{rel_path}"
                        
                        # 提取Blend文件路径
                        if "blend_file_path" in data:
                            blend_path = data["blend_file_path"]
                            if os.path.exists(blend_path):
                                rel_path = os.path.relpath(blend_path, settings.OUTPUT_DIR)
                                result.blend_file_url = f"/outputs/{rel_path}"
                        elif "script_path" in data:
                            # 如果脚本路径是.blend文件
                            script_path = data["script_path"]
                            if script_path.endswith('.blend') and os.path.exists(script_path):
                                rel_path = os.path.relpath(script_path, settings.OUTPUT_DIR)
                                result.blend_file_url = f"/outputs/{rel_path}"
                        
                        # 提取渲染图
                        if "rendered_images" in data:
                            rendered = data["rendered_images"]
                            if isinstance(rendered, dict):
                                for view_name, img_path in rendered.items():
                                    if os.path.exists(img_path):
                                        rel_path = os.path.relpath(img_path, settings.OUTPUT_DIR)
                                        result.render_views.append(f"/outputs/{rel_path}")
                
                except:
                    continue
        
        except Exception as e:
            print(f"提取生成结果失败: {e}")
        
        return result
    
    def cancel_task(self, session_id: str) -> bool:
        """
        取消指定任务
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功标记为取消
        """
        if session_id in self._running_tasks:
            self._running_tasks[session_id] = True
            print(f"🛑 [TaskExecutor] 标记任务取消: {session_id}")
            return True
        else:
            return False
    
    def cancel_all_tasks(self) -> int:
        """
        取消所有正在运行的任务
        
        Returns:
            int: 被取消的任务数量
        """
        count = 0
        for session_id in list(self._running_tasks.keys()):
            self._running_tasks[session_id] = True
            count += 1
            print(f"🛑 [TaskExecutor] 标记任务取消: {session_id}")
        
        if count > 0:
            print(f"🛑 [TaskExecutor] 已标记取消 {count} 个任务")
        else:
            print(f"ℹ️ [TaskExecutor] 没有正在运行的任务")
        
        return count
    
    def get_running_tasks(self, include_cancelling: bool = False) -> List[str]:
        """
        获取正在运行的任务列表
        
        Args:
            include_cancelling: 是否包含正在取消的任务（默认不包含）
            
        Returns:
            任务ID列表
        """
        if include_cancelling:
            return list(self._running_tasks.keys())
        else:
            # 只返回未标记取消的任务（值为False的任务）
            return [task_id for task_id, is_cancelled in self._running_tasks.items() if not is_cancelled]
    
    def get_cancelling_tasks(self) -> List[str]:
        """获取正在取消中的任务列表"""
        return [task_id for task_id, is_cancelled in self._running_tasks.items() if is_cancelled]
    
    def shutdown(self):
        """关闭执行器"""
        print("🛑 [TaskExecutor] 关闭执行器...")
        self.executor.shutdown(wait=True)


# 全局单例
_task_executor: Optional[TaskExecutor] = None


def get_task_executor() -> TaskExecutor:
    """获取任务执行器单例"""
    global _task_executor
    if _task_executor is None:
        _task_executor = TaskExecutor()
    return _task_executor
