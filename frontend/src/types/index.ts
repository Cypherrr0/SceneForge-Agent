/**
 * 类型定义
 */

export type SessionStatus =
  | 'idle'
  | 'parsing'
  | 'planning'
  | 'executing'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type StepStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'skipped';

export interface TaskStep {
  step_id: string;
  name: string;
  description: string;
  tool: string;
  status: StepStatus;
  progress: number;
  start_time?: string;
  end_time?: string;
  result?: any;
  error?: string;
}

export interface GenerationIntent {
  generation_prompt: string;
  wants: Record<string, boolean>;
  constraints: Record<string, boolean>;
  rendering_complexity: Record<string, any>;
  reason: string;
}

export interface GenerationResult {
  optimized_prompt?: string;
  image_2d_url?: string;
  model_3d_url?: string;
  render_views: string[];
  blend_file_url?: string;
  video_url?: string;
  quality_score?: number;
  evaluation_details?: any;
}

export interface Session {
  session_id: string;
  user_prompt: string;
  status: SessionStatus;
  intent?: GenerationIntent;
  plan?: any;
  steps: TaskStep[];
  current_step_index: number;
  total_steps: number;
  overall_progress: number;
  result: GenerationResult;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  error_details?: any;
  interactive: boolean;
  waiting_for_user: boolean;
  output_dir: string;
}

// API请求/响应类型
export interface CreateSessionRequest {
  prompt: string;
  interactive?: boolean;
}

export interface CreateSessionResponse {
  success: boolean;
  message?: string;
  session: Session;
}

export interface StartGenerationRequest {
  session_id: string;
}

export interface GenerationStatusResponse {
  success: boolean;
  session_id: string;
  status: SessionStatus;
  current_step?: string;
  progress: number;
  total_steps: number;
  completed_steps: number;
  estimated_time_remaining?: number;
}

export interface GenerationResultResponse {
  success: boolean;
  session_id: string;
  status: SessionStatus;
  result: GenerationResult;
  execution_time?: number;
}

// WebSocket消息类型
export type WebSocketMessageType =
  | 'connected'
  | 'progress'
  | 'step_complete'
  | 'status_update'
  | 'completed'
  | 'error'
  | 'cancelled'
  | 'llm_message'
  | 'user_input_required'
  | 'log_message';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  timestamp: string;
  [key: string]: any;
}

