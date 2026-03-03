/**
 * 生成任务状态管理
 */
import { create } from 'zustand';
import type { Session, WebSocketMessage } from '@/types';

interface GenerationStore {
  // 当前会话
  currentSession: Session | null;
  setCurrentSession: (session: Session | null) => void;

  // WebSocket连接状态
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // 实时消息
  messages: WebSocketMessage[];
  addMessage: (message: WebSocketMessage) => void;
  clearMessages: () => void;

  // 加载状态
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // 错误状态
  error: string | null;
  setError: (error: string | null) => void;

  // 重置状态
  reset: () => void;
}

export const useGenerationStore = create<GenerationStore>((set) => ({
  currentSession: null,
  setCurrentSession: (session) => set({ currentSession: session }),

  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),

  messages: [],
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  clearMessages: () => set({ messages: [] }),

  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  error: null,
  setError: (error) => set({ error }),

  reset: () =>
    set({
      currentSession: null,
      wsConnected: false,
      messages: [],
      isLoading: false,
      error: null,
    }),
}));

