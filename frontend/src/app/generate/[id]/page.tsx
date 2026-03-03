'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { sessionApi } from '@/services/api';
import { getWebSocketService } from '@/services/websocket';
import type { Session, WebSocketMessage } from '@/types';
import { Loader2, CheckCircle2, XCircle, ArrowLeft, ChevronLeft, ChevronRight, Send, Image as ImageIcon, Box, FileCode } from 'lucide-react';

interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
}

type ViewerMode = 'image' | 'model' | 'blend';

export default function GeneratePage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.id as string;
  
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  
  // 聊天相关状态
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [userInput, setUserInput] = useState('');
  const [waitingForResponse, setWaitingForResponse] = useState(false);
  const [waitingForInteraction, setWaitingForInteraction] = useState(false);
  const [showReplanInput, setShowReplanInput] = useState(false); // 显示重新规划输入框
  const [replanInput, setReplanInput] = useState(''); // 重新规划的输入内容
  const chatEndRef = useRef<HTMLDivElement>(null);
  const messageIdCounter = useRef(0); // 用于生成唯一的消息ID
  
  // 展示框相关状态
  const [viewerMode, setViewerMode] = useState<ViewerMode>('image');
  const [availableAssets, setAvailableAssets] = useState({
    image: null as string | null,
    model: null as string | null,
    blend: null as string | null,
  });
  
  // 使用 ref 确保绝对只执行一次
  const hasInitialized = useRef(false);

  // 自动滚动到聊天底部
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // WebSocket 连接
  useEffect(() => {
    const ws = getWebSocketService();
    
    // 消息处理器
    const handleMessage = (msg: WebSocketMessage) => {
      console.log('收到WebSocket消息:', msg);
      setMessages(prev => [...prev, msg]);
      
      // 根据消息类型更新状态
      if (msg.type === 'status_update' && msg.session) {
        setSession(msg.session);
        
        // 更新可用资源
        if (msg.session.result) {
          console.log('更新资源 (status_update):', msg.session.result);
          setAvailableAssets(prev => {
            const newAssets = {
              image: msg.session.result.image_2d_url || prev.image,
              model: msg.session.result.model_3d_url || prev.model,
              blend: msg.session.result.blend_file_url || prev.blend,
            };
            console.log('新资源状态:', newAssets);
            console.log('当前 viewerMode:', viewerMode);
            return newAssets;
          });
        }
      } else if (msg.type === 'completed') {
        setSession(msg.session);
        setWaitingForResponse(false);
        
        // 更新可用资源
        if (msg.session && msg.session.result) {
          console.log('更新资源 (completed):', msg.session.result);
          setAvailableAssets(prev => {
            const newAssets = {
              image: msg.session.result.image_2d_url || prev.image,
              model: msg.session.result.model_3d_url || prev.model,
              blend: msg.session.result.blend_file_url || prev.blend,
            };
            console.log('新资源状态:', newAssets);
            return newAssets;
          });
        }
      } else if (msg.type === 'error') {
        setError(msg.message || '生成失败');
        setWaitingForResponse(false);
      } else if (msg.type === 'llm_message') {
        // LLM 对话消息
        messageIdCounter.current += 1;
        setChatMessages(prev => [...prev, {
          id: `agent-msg-${messageIdCounter.current}`,
          type: 'agent',
          content: msg.content || msg.message || '',
          timestamp: new Date(msg.timestamp),
        }]);
        setWaitingForResponse(false);
      } else if (msg.type === 'user_input_required') {
        // 需要用户输入 - 显示交互按钮
        messageIdCounter.current += 1;
        setChatMessages(prev => [...prev, {
          id: `agent-msg-${messageIdCounter.current}`,
          type: 'agent',
          content: msg.message || msg.prompt || '请选择操作',
          timestamp: new Date(msg.timestamp),
        }]);
        setWaitingForResponse(false);
        setWaitingForInteraction(true);  // 显示交互按钮
        
        // 更新可用资源（如果消息包含session）
        if (msg.session && msg.session.result) {
          console.log('更新资源 (user_input_required):', msg.session.result);
          setAvailableAssets(prev => {
            const newAssets = {
              image: msg.session.result.image_2d_url || prev.image,
              model: msg.session.result.model_3d_url || prev.model,
              blend: msg.session.result.blend_file_url || prev.blend,
            };
            console.log('新资源状态:', newAssets);
            return newAssets;
          });
        }
      }
    };

    const handleConnected = () => {
      console.log('WebSocket已连接');
      setWsConnected(true);
    };

    const handleDisconnected = () => {
      console.log('WebSocket已断开');
      setWsConnected(false);
    };

    // 注册事件监听
    ws.on('connected', handleConnected);
    ws.on('disconnected', handleDisconnected);
    ws.on('progress', handleMessage);
    ws.on('status_update', handleMessage);
    ws.on('completed', handleMessage);
    ws.on('error', handleMessage);
    ws.on('llm_message', handleMessage);
    ws.on('user_input_required', handleMessage);
    ws.on('log_message', handleMessage);
    // 连接WebSocket
    ws.connect(sessionId);

    // 清理
    return () => {
      ws.off('connected', handleConnected);
      ws.off('disconnected', handleDisconnected);
      ws.off('progress', handleMessage);
      ws.off('status_update', handleMessage);
      ws.off('completed', handleMessage);
      ws.off('error', handleMessage);
      ws.off('llm_message', handleMessage);
      ws.off('user_input_required', handleMessage);
      ws.off('log_message', handleMessage);
      ws.disconnect();
    };
  }, [sessionId]);

  // 获取初始会话信息并启动任务（只执行一次）
  useEffect(() => {
    // 使用 ref 确保绝对只执行一次
    if (hasInitialized.current) {
      return;
    }
    hasInitialized.current = true;
    
    const init = async () => {
      try {
        console.log('🔄 初始化会话:', sessionId);
        
        // 获取会话信息
        const statusResponse = await sessionApi.getStatus(sessionId);
        console.log('📊 会话状态:', statusResponse.status);
        
        // 只有在 idle 状态才启动任务
        if (statusResponse.status === 'idle') {
          console.log('🚀 启动生成任务...');
          
          try {
            const response = await sessionApi.startGeneration({ session_id: sessionId });
            console.log('✅ 生成任务已启动:', response);
            setIsInitializing(false);
          } catch (err: any) {
            console.error('❌ 启动生成失败:', err);
            setError('启动生成失败: ' + (err.response?.data?.detail || err.message));
            setIsInitializing(false);
          }
        } else if (statusResponse.status === 'failed') {
          console.log('⚠️ 会话已失败');
          setError('会话已失败');
          setIsInitializing(false);
        } else {
          // parsing, planning, executing, completed 等状态
          console.log('✅ 会话已在运行，状态:', statusResponse.status);
          setIsInitializing(false);
        }
      } catch (err: any) {
        console.error('❌ 初始化失败:', err);
        setError('获取会话信息失败: ' + (err.response?.data?.detail || err.message));
        setIsInitializing(false);
      }
    };

    init();
  }, [sessionId]); // 只依赖 sessionId

  // 处理用户交互选择
  const handleInteractionChoice = async (action: 'continue' | 'stop' | 'replan') => {
    try {
      // 如果是重新规划，显示输入框
      if (action === 'replan') {
        setShowReplanInput(true);
        return;
      }
      
      setWaitingForInteraction(false);
      setWaitingForResponse(true);
      
      // 添加用户选择到聊天
      const actionText = {
        'continue': '继续下一步 ✓',
        'stop': '停止执行 ✗',
      }[action];
      
      messageIdCounter.current += 1;
      setChatMessages(prev => [...prev, {
        id: `user-msg-${messageIdCounter.current}`,
        type: 'user',
        content: actionText,
        timestamp: new Date(),
      }]);
      
      // 发送交互响应到后端
      await sessionApi.sendInteraction(sessionId, action);
      
    } catch (err) {
      console.error('发送交互失败:', err);
      setWaitingForResponse(false);
      setWaitingForInteraction(true);
    }
  };
  
  // 处理重新规划的提交
  const handleReplanSubmit = async () => {
    if (!replanInput.trim()) return;
    
    try {
      setShowReplanInput(false);
      setWaitingForInteraction(false);
      setWaitingForResponse(true);
      
      // 添加用户输入到聊天
      messageIdCounter.current += 1;
      setChatMessages(prev => [...prev, {
        id: `user-msg-${messageIdCounter.current}`,
        type: 'user',
        content: `重新规划: ${replanInput}`,
        timestamp: new Date(),
      }]);
      
      // 发送重新规划请求到后端（带上新需求）
      await sessionApi.sendInteraction(sessionId, 'replan', replanInput);
      
      // 清空输入
      setReplanInput('');
      
    } catch (err) {
      console.error('发送重新规划失败:', err);
      setWaitingForResponse(false);
      setWaitingForInteraction(true);
      setShowReplanInput(false);
    }
  };
  
  // 取消重新规划，返回按钮组
  const handleCancelReplan = () => {
    setShowReplanInput(false);
    setReplanInput('');
  };
  
  // 发送用户消息（暂时未使用）
  const handleSendMessage = async () => {
    if (!userInput.trim() || waitingForResponse) return;
    
    const message = userInput.trim();
    setUserInput('');
    setWaitingForResponse(true);
    
    // 添加用户消息到聊天
    messageIdCounter.current += 1;
    setChatMessages(prev => [...prev, {
      id: `user-msg-${messageIdCounter.current}`,
      type: 'user',
      content: message,
      timestamp: new Date(),
    }]);
    
    try {
      // TODO: 支持自由文本输入
      console.log('发送用户输入:', message);
    } catch (err) {
      console.error('发送失败:', err);
      setWaitingForResponse(false);
    }
  };

  // 切换展示模式
  const switchViewerMode = (direction: 'prev' | 'next') => {
    const modes: ViewerMode[] = ['image', 'model', 'blend'];
    const currentIndex = modes.indexOf(viewerMode);
    
    if (direction === 'prev') {
      const newIndex = (currentIndex - 1 + modes.length) % modes.length;
      setViewerMode(modes[newIndex]);
    } else {
      const newIndex = (currentIndex + 1) % modes.length;
      setViewerMode(modes[newIndex]);
    }
  };

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        switchViewerMode('prev');
      } else if (e.key === 'ArrowRight') {
        switchViewerMode('next');
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [viewerMode]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* 头部 */}
      <div className="bg-slate-900/95 border-b border-slate-700 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            返回首页
          </button>
          
          <div className="flex items-center gap-4">
            {wsConnected ? (
              <div className="flex items-center gap-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-sm">实时连接</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-gray-500">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span className="text-sm">连接中...</span>
              </div>
            )}
            
            {/* 终止按钮 */}
            <button
              onClick={async () => {
                if (confirm('确定要终止当前任务吗？')) {
                  try {
                    await sessionApi.cancelSession(sessionId);
                  } catch (err) {
                    console.error('终止失败:', err);
                  }
                }
              }}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
            >
              <XCircle className="w-4 h-4" />
              终止任务
            </button>
          </div>
        </div>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="bg-red-500/10 border-b border-red-500/30 p-4">
          <div className="max-w-7xl mx-auto flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-red-400 font-medium mb-1">错误</h3>
              <p className="text-sm text-gray-300">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* 主内容区 */}
      <div className="flex overflow-hidden h-[calc(100vh-80px)]">
        <div className="max-w-7xl mx-auto w-full flex gap-4 p-4">
          {/* 左侧区域：进度 + 聊天框 (2/3) */}
          <div className="flex-[2] flex flex-col gap-4 min-w-0 h-full">
            {/* 生成进度 */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700 flex-shrink-0">
              <h2 className="text-xl font-bold mb-4 text-white">生成进度</h2>
              
              <div className="space-y-3">
                {isInitializing && (
                  <div className="flex items-center gap-3 text-blue-400">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>正在初始化...</span>
                  </div>
                )}
                
                {session && (
                  <>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">状态:</span>
                      <span className="text-white font-medium">{session.status}</span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">进度:</span>
                      <span className="text-white font-medium">{session.overall_progress}%</span>
                    </div>
                    
                    {/* 进度条 */}
                    <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                        style={{ width: `${session.overall_progress}%` }}
                      />
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* 聊天框 */}
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700 flex flex-col flex-1 min-h-0">
              <div className="p-4 border-b border-slate-700 flex-shrink-0">
                <h3 className="text-lg font-bold text-white">对话</h3>
              </div>
              
              {/* 消息列表 */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatMessages.length === 0 ? (
                  <div className="text-center text-gray-500 mt-8">
                    <p>暂无对话消息</p>
                  </div>
                ) : (
                  (() => {
                    // 将连续的Agent消息分组
                    const messageGroups: Array<{type: 'user' | 'agent', messages: ChatMessage[]}> = [];
                    chatMessages.forEach((msg) => {
                      const lastGroup = messageGroups[messageGroups.length - 1];
                      if (lastGroup && lastGroup.type === msg.type) {
                        // 同类型消息，加入当前组
                        lastGroup.messages.push(msg);
                      } else {
                        // 新类型消息，创建新组
                        messageGroups.push({ type: msg.type, messages: [msg] });
                      }
                    });

                    return messageGroups.map((group, groupIdx) => (
                      <div key={`group-${groupIdx}`}>
                        {group.type === 'user' ? (
                          /* 用户消息组 - 每条独立显示 */
                          group.messages.map((msg) => (
                            <div key={msg.id} className="py-2 mb-3">
                              <div className="flex items-baseline gap-2 mb-2">
                                <span className="text-xs text-gray-500">
                                  {msg.timestamp.toLocaleTimeString()}
                                </span>
                              </div>
                              <div className="bg-blue-600/20 rounded-lg px-4 py-3 border border-blue-500/30">
                                <p className="text-sm text-gray-100 whitespace-pre-wrap break-words">
                                  {msg.content}
                                </p>
                              </div>
                            </div>
                          ))
                        ) : (
                          /* Agent消息组 - 合并显示 */
                          <div className="py-2">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className="text-xs text-gray-500">
                                {group.messages[0].timestamp.toLocaleTimeString()}
                              </span>
                            </div>
                            {/* 紧凑排列的消息内容，无边框 */}
                            <div className="space-y-2">
                              {group.messages.map((msg) => (
                                <div key={msg.id}>
                                  <p className="text-sm text-gray-200 whitespace-pre-wrap break-words leading-relaxed">
                                    {msg.content}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ));
                  })()
                )}
                <div ref={chatEndRef} />
              </div>
              
              {/* 输入区域 */}
              <div className="p-4 border-t border-slate-700 flex-shrink-0">
                {waitingForInteraction ? (
                  showReplanInput ? (
                    /* 重新规划输入框 */
                    <div className="flex flex-col gap-3">
                      <div className="flex items-center gap-2 mb-1">
                        <button
                          onClick={handleCancelReplan}
                          className="p-1.5 hover:bg-slate-700 rounded-lg transition-colors"
                          title="返回"
                        >
                          <ArrowLeft className="w-4 h-4 text-gray-400" />
                        </button>
                        <p className="text-sm text-gray-400">请输入新的需求：</p>
                      </div>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={replanInput}
                          onChange={(e) => setReplanInput(e.target.value)}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                              e.preventDefault();
                              handleReplanSubmit();
                            }
                          }}
                          placeholder="例如：将花瓶改成蓝色..."
                          autoFocus
                          className="flex-1 bg-slate-700 border border-slate-600 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                        />
                        <button
                          onClick={handleReplanSubmit}
                          disabled={!replanInput.trim()}
                          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 font-medium"
                        >
                          <Send className="w-5 h-5" />
                          提交
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* 交互按钮 */
                    <div className="flex flex-col gap-2">
                      <p className="text-sm text-gray-400 mb-2">请选择操作：</p>
                      <div className="grid grid-cols-3 gap-2">
                        <button
                          onClick={() => handleInteractionChoice('continue')}
                          disabled={waitingForResponse}
                          className="px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
                        >
                          <CheckCircle2 className="w-5 h-5" />
                          继续
                        </button>
                        <button
                          onClick={() => handleInteractionChoice('stop')}
                          disabled={waitingForResponse}
                          className="px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
                        >
                          <XCircle className="w-5 h-5" />
                          停止
                        </button>
                        <button
                          onClick={() => handleInteractionChoice('replan')}
                          disabled={waitingForResponse}
                          className="px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
                        >
                          <Loader2 className="w-5 h-5" />
                          重新规划
                        </button>
                      </div>
                    </div>
                  )
                ) : (
                  /* 普通输入框（暂时禁用） */
                  <div className="flex gap-2 opacity-50">
                    <input
                      type="text"
                      value={userInput}
                      onChange={(e) => setUserInput(e.target.value)}
                      placeholder="等待Agent响应..."
                      disabled={true}
                      className="flex-1 bg-slate-700 border border-slate-600 rounded-xl px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 disabled:opacity-50 cursor-not-allowed"
                    />
                    <button
                      disabled={true}
                      className="px-4 py-2 bg-slate-600 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <Send className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 右侧区域：展示框 + 实时日志 (1/3) */}
          <div className="flex-1 flex flex-col gap-4 min-w-0 h-full">
            {/* 展示框 */}
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700 flex flex-col h-[55%] flex-shrink-0">
              <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                <h3 className="text-lg font-bold text-white">预览</h3>
                
                {/* 模式切换按钮 */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => switchViewerMode('prev')}
                    className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <ChevronLeft className="w-5 h-5 text-gray-400" />
                  </button>
                  
                  <div className="flex gap-1">
                    <button
                      onClick={() => setViewerMode('image')}
                      className={`p-2 rounded-lg transition-colors ${
                        viewerMode === 'image' ? 'bg-blue-600 text-white' : 'hover:bg-slate-700 text-gray-400'
                      }`}
                      title="图片"
                    >
                      <ImageIcon className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setViewerMode('model')}
                      className={`p-2 rounded-lg transition-colors ${
                        viewerMode === 'model' ? 'bg-blue-600 text-white' : 'hover:bg-slate-700 text-gray-400'
                      }`}
                      title="3D模型"
                    >
                      <Box className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setViewerMode('blend')}
                      className={`p-2 rounded-lg transition-colors ${
                        viewerMode === 'blend' ? 'bg-blue-600 text-white' : 'hover:bg-slate-700 text-gray-400'
                      }`}
                      title="Blend文件"
                    >
                      <FileCode className="w-4 h-4" />
                    </button>
                  </div>
                  
                  <button
                    onClick={() => switchViewerMode('next')}
                    className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>
              
              {/* 展示内容 */}
              <div className="flex-1 p-4 flex items-center justify-center overflow-hidden">
                {(() => {
                  console.log('渲染预览区域 - viewerMode:', viewerMode, 'availableAssets:', availableAssets);
                  return null;
                })()}
                {viewerMode === 'image' && availableAssets.image ? (
                  <img
                    src={availableAssets.image}
                    alt="Generated"
                    className="max-w-full max-h-full object-contain rounded-lg"
                    onLoad={() => console.log('图片加载成功:', availableAssets.image)}
                    onError={(e) => console.error('图片加载失败:', availableAssets.image, e)}
                  />
                ) : viewerMode === 'model' && availableAssets.model ? (
                  <div className="w-full h-full">
                    <model-viewer
                      src={availableAssets.model}
                      alt="3D Model"
                      auto-rotate
                      camera-controls
                      style={{ width: '100%', height: '100%' }}
                    />
                  </div>
                ) : viewerMode === 'blend' && availableAssets.blend ? (
                  <div className="text-center">
                    <FileCode className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-400 mb-4">Blend文件已生成</p>
                    <a
                      href={availableAssets.blend}
                      download
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg inline-block"
                    >
                      下载文件
                    </a>
                  </div>
                ) : (
                  <div className="text-center text-gray-500">
                    <p>暂无{viewerMode === 'image' ? '图片' : viewerMode === 'model' ? '3D模型' : 'Blend文件'}</p>
                  </div>
                )}
              </div>
            </div>

            {/* 实时日志 */}
            <div className="bg-slate-800/50 rounded-2xl p-4 border border-slate-700 flex-1 overflow-hidden flex flex-col min-h-0">
              <h3 className="text-sm font-bold mb-3 text-white flex-shrink-0">实时日志</h3>
              
              <div className="flex-1 overflow-y-auto space-y-1 text-xs font-mono">
                {messages.length === 0 ? (
                  <p className="text-gray-500">等待日志...</p>
                ) : (
                  messages.map((msg, idx) => {
                    // 提取消息内容
                    const content = msg.message || msg.content || msg.prompt || '';
                    const displayType = msg.type === 'llm_message' ? 'LLM' : 
                                       msg.type === 'log_message' ? 'LOG' :
                                       msg.type === 'user_input_required' ? '等待输入' :
                                       msg.type === 'status_update' ? '状态' :
                                       msg.type === 'progress' ? '进度' :
                                       msg.type === 'completed' ? '完成' :
                                       msg.type === 'error' ? '错误' :
                                       msg.type;
                    
                    return (
                      <div key={idx} className="text-gray-400 leading-relaxed">
                        <span className="text-gray-600">[{new Date(msg.timestamp).toLocaleTimeString()}]</span>
                        <span className={`ml-2 font-semibold ${
                          msg.type === 'error' ? 'text-red-400' :
                          msg.type === 'completed' ? 'text-green-400' :
                          msg.type === 'user_input_required' ? 'text-yellow-400' :
                          msg.type === 'log_message' ? 'text-gray-500' :
                          'text-blue-400'
                        }`}>[{displayType}]</span>
                        {content && <span className="ml-2 text-gray-300">{content}</span>}
                        {msg.progress !== undefined && (
                          <span className="ml-2 text-gray-300">进度: {msg.progress}%</span>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

