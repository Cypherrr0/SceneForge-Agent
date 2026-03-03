/**
 * API服务
 */
import axios from 'axios';
import type {
  CreateSessionRequest,
  CreateSessionResponse,
  StartGenerationRequest,
  GenerationStatusResponse,
  GenerationResultResponse,
  Session,
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2分钟超时
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('Network Error:', error.message);
    }
    return Promise.reject(error);
  }
);

export const sessionApi = {
  /**
   * 创建新会话
   */
  async createSession(data: CreateSessionRequest): Promise<CreateSessionResponse> {
    const response = await api.post<CreateSessionResponse>('/sessions/create', data);
    return response.data;
  },

  /**
   * 开始生成任务
   */
  async startGeneration(data: StartGenerationRequest): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/sessions/start', data);
    return response.data;
  },

  /**
   * 获取生成状态
   */
  async getStatus(sessionId: string): Promise<GenerationStatusResponse> {
    const response = await api.get<GenerationStatusResponse>(`/sessions/status/${sessionId}`);
    return response.data;
  },

  /**
   * 获取生成结果
   */
  async getResult(sessionId: string): Promise<GenerationResultResponse> {
    const response = await api.get<GenerationResultResponse>(`/sessions/result/${sessionId}`);
    return response.data;
  },

  /**
   * 列出会话
   */
  async listSessions(limit = 50, offset = 0): Promise<{ sessions: any[]; total: number }> {
    const response = await api.get(`/sessions/list?limit=${limit}&offset=${offset}`);
    return response.data;
  },

  /**
   * 取消会话
   */
  async cancelSession(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/sessions/cancel', { session_id: sessionId });
    return response.data;
  },

  /**
   * 取消所有会话
   */
  async cancelAllSessions(): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/sessions/cancel-all');
    return response.data;
  },

  /**
   * 发送用户交互响应
   */
  async sendInteraction(sessionId: string, action: 'continue' | 'stop' | 'replan', message?: string): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/sessions/interact', {
      session_id: sessionId,
      action,
      message
    });
    return response.data;
  },
};

export default api;

