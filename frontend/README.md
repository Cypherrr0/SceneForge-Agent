# Hunyuan3D Agent Frontend

企业级3D生成前端应用，基于Next.js 14 + TypeScript + Tailwind CSS。

## 技术栈

- ⚡ **Next.js 14**: App Router架构
- 🎨 **TypeScript**: 类型安全
- 💅 **Tailwind CSS**: 实用优先的CSS框架
- 🎭 **Framer Motion**: 动画库
- 🔮 **Three.js**: 3D渲染
- 📡 **WebSocket**: 实时通信
- 🗂️ **Zustand**: 轻量级状态管理
- 🔄 **React Query**: 数据获取和缓存

## 目录结构

```
frontend/
├── src/
│   ├── app/                    # Next.js 14 App Router
│   │   ├── page.tsx            # 首页
│   │   ├── layout.tsx          # 根布局
│   │   ├── globals.css         # 全局样式
│   │   └── generate/           # 生成页面
│   │       └── [id]/
│   │           └── page.tsx    # 动态生成页
│   ├── components/             # 组件
│   │   ├── ui/                 # 基础UI组件
│   │   └── features/           # 功能组件
│   │       ├── ProgressTracker.tsx    # 进度追踪器
│   │       ├── ModelViewer.tsx        # 3D模型查看器
│   │       └── ResultGallery.tsx      # 结果展示
│   ├── lib/                    # 工具函数
│   │   └── utils.ts
│   ├── services/               # API服务
│   │   ├── api.ts              # REST API
│   │   └── websocket.ts        # WebSocket
│   ├── store/                  # 状态管理
│   │   └── generation-store.ts
│   ├── hooks/                  # 自定义Hooks
│   │   └── useWebSocket.ts
│   └── types/                  # TypeScript类型
│       └── index.ts
├── public/                     # 静态资源
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── next.config.js
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 环境变量

创建 `.env.local` 文件：

```env
NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8080/api/v1
```

### 3. 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:3000

### 4. 构建生产版本

```bash
npm run build
npm start
```

## 核心功能

### 1. 首页 (/)

- ✅ 优雅的Hero设计
- ✅ 提示词输入框
- ✅ 示例提示词快速选择
- ✅ 实时字符计数
- ✅ 加载状态处理

### 2. 生成页面 (/generate/[id])

**实时进度追踪**
- WebSocket实时连接
- 步骤进度可视化
- 当前状态显示
- 预计剩余时间

**中间结果预览**
- 优化后的提示词
- 生成的2D图像
- 实时日志输出

**3D模型查看器**
- 交互式3D预览
- 旋转、缩放、平移
- 材质查看
- 多视角切换

**结果展示**
- 生成结果Gallery
- 多视角渲染图
- 质量评分显示
- 一键下载

### 3. UI组件

**基础组件**
- Button: 按钮组件
- Card: 卡片容器
- Progress: 进度条
- Badge: 标签
- Dialog: 对话框
- Tabs: 标签页

**功能组件**
- ProgressTracker: 步骤进度追踪
- ModelViewer: 3D模型查看器
- ResultGallery: 结果展示画廊
- StatusIndicator: 状态指示器

## API集成

### REST API

```typescript
import { sessionApi } from '@/services/api';

// 创建会话
const response = await sessionApi.createSession({
  prompt: '一只可爱的猫咪',
  interactive: false,
});

// 开始生成
await sessionApi.startGeneration({
  session_id: response.session.session_id,
});

// 查询状态
const status = await sessionApi.getStatus(sessionId);

// 获取结果
const result = await sessionApi.getResult(sessionId);
```

### WebSocket

```typescript
import { getWebSocketService } from '@/services/websocket';

const ws = getWebSocketService();

// 连接
ws.connect(sessionId);

// 监听消息
ws.on('progress', (message) => {
  console.log('进度更新:', message.progress);
});

ws.on('completed', (message) => {
  console.log('任务完成:', message.result);
});

// 断开连接
ws.disconnect();
```

## 状态管理

使用Zustand进行轻量级状态管理：

```typescript
import { useGenerationStore } from '@/store/generation-store';

function Component() {
  const { currentSession, setCurrentSession } = useGenerationStore();
  
  // 使用状态...
}
```

## 样式系统

### Tailwind CSS配置

- 深色主题优先
- 自定义颜色变量
- 响应式设计
- 动画效果

### 自定义样式

```tsx
// 玻璃态效果
<div className="glass rounded-xl p-4">
  Content
</div>

// 渐变背景
<div className="gradient-bg">
  Content
</div>

// 状态颜色
<span className={getStatusColor(status)}>
  {getStatusLabel(status)}
</span>
```

## 性能优化

1. **代码分割**
   - 动态导入
   - 路由级代码分割

2. **图像优化**
   - Next.js Image组件
   - 懒加载
   - WebP格式

3. **3D模型优化**
   - LOD（多细节层次）
   - 延迟加载
   - 模型压缩

4. **缓存策略**
   - React Query缓存
   - Service Worker
   - 浏览器缓存

## 部署

### Vercel部署（推荐）

```bash
npm install -g vercel
vercel
```

### Docker部署

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["npm", "start"]
```

### Nginx配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket支持
    location /api/v1/ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 开发指南

### 添加新页面

```tsx
// src/app/new-page/page.tsx
export default function NewPage() {
  return <div>New Page</div>;
}
```

### 添加新组件

```tsx
// src/components/features/NewComponent.tsx
export function NewComponent() {
  return <div>New Component</div>;
}
```

### 添加新Hook

```tsx
// src/hooks/useNewHook.ts
export function useNewHook() {
  // Hook logic
}
```

## 故障排查

### 1. WebSocket连接失败

检查：
- 后端服务是否运行
- WebSocket URL是否正确
- CORS配置是否正确

### 2. 3D模型加载失败

检查：
- 模型文件路径是否正确
- 文件格式是否支持（GLB/GLTF）
- 文件大小是否过大

### 3. 样式不生效

检查：
- Tailwind配置是否正确
- PostCSS配置是否正确
- 是否需要重启开发服务器

## 浏览器支持

- Chrome/Edge: 最新版本
- Firefox: 最新版本
- Safari: 最新版本
- 移动端浏览器: 现代浏览器

## License

MIT

