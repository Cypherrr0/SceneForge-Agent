# Hunyuan3D Agent Backend

企业级3D生成后端服务，基于FastAPI + WebSocket实现实时通信。

## 架构特点

- ✅ **RESTful API**: 标准的REST接口设计
- ✅ **WebSocket实时通信**: 实时进度推送
- ✅ **异步任务处理**: 后台执行长任务
- ✅ **会话管理**: 完整的会话生命周期管理
- ✅ **类型安全**: 使用Pydantic进行数据验证
- ✅ **企业级代码结构**: 分层架构，易于维护

## 目录结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI主应用
│   ├── core/                   # 核心配置
│   │   ├── __init__.py
│   │   └── config.py
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   └── session.py
│   ├── schemas/                # API Schema
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   └── responses.py
│   ├── services/               # 业务逻辑
│   │   ├── __init__.py
│   │   ├── session_service.py  # 会话管理
│   │   ├── task_executor.py    # 任务执行器
│   │   └── websocket_manager.py # WebSocket管理
│   └── api/                    # API路由
│       └── v1/
│           ├── __init__.py
│           ├── sessions.py     # 会话API
│           └── websocket.py    # WebSocket端点
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 3. 启动服务

```bash
# 开发模式（自动重载）
python -m app.main

# 或使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 4. 访问API文档

打开浏览器访问：
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## API接口

### 会话管理

#### 创建会话
```http
POST /api/v1/sessions/create
Content-Type: application/json

{
  "prompt": "一只可爱的猫咪坐在椅子上",
  "interactive": false
}
```

#### 开始生成
```http
POST /api/v1/sessions/start
Content-Type: application/json

{
  "session_id": "session_20231120_123456_abc123"
}
```

#### 查询状态
```http
GET /api/v1/sessions/status/{session_id}
```

#### 获取结果
```http
GET /api/v1/sessions/result/{session_id}
```

#### 取消任务
```http
POST /api/v1/sessions/cancel
Content-Type: application/json

{
  "session_id": "session_20231120_123456_abc123"
}
```

### WebSocket连接

```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/ws/{session_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'connected':
      console.log('WebSocket连接成功');
      break;
    case 'progress':
      console.log(`进度: ${data.progress}%`);
      break;
    case 'step_complete':
      console.log(`步骤完成: ${data.step_name}`);
      break;
    case 'completed':
      console.log('任务完成', data.result);
      break;
    case 'error':
      console.error('错误', data.error);
      break;
  }
};

// 心跳
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
  }
}, 30000);
```

## 开发指南

### 添加新的API端点

1. 在 `app/schemas/requests.py` 定义请求Schema
2. 在 `app/schemas/responses.py` 定义响应Schema
3. 在 `app/api/v1/` 创建新的路由文件
4. 在 `app/main.py` 注册路由

### 添加新的服务

1. 在 `app/services/` 创建服务文件
2. 实现业务逻辑
3. 在 `app/services/__init__.py` 导出服务

## 部署

### 生产环境配置

```bash
# 使用gunicorn + uvicorn workers
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080 \
  --timeout 120
```

### Docker部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## 性能优化

- 使用异步IO处理并发请求
- 后台任务队列处理长时间任务
- WebSocket连接池管理
- 会话自动清理机制
- 文件静态服务器优化

## 安全考虑

- API密钥环境变量隔离
- CORS白名单配置
- 请求参数验证
- 文件大小限制
- 会话超时机制

