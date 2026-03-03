'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2, Sparkles, Box } from 'lucide-react';
import { sessionApi } from '@/services/api';
import { useGenerationStore } from '@/store/generation-store';

export default function HomePage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const setCurrentSession = useGenerationStore((state) => state.setCurrentSession);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    try {
      setIsLoading(true);

      console.log('开始创建会话，提示词:', prompt.trim());

      // 创建会话（默认开启交互模式）
      const response = await sessionApi.createSession({
        prompt: prompt.trim(),
        interactive: true,
      });

      console.log('创建会话响应:', response);

      if (response.success && response.session) {
        setCurrentSession(response.session);
        
        console.log('会话创建成功，跳转到生成页面:', response.session.session_id);
        // 跳转到生成页面
        router.push(`/generate/${response.session.session_id}`);
      } else {
        console.error('会话创建失败，响应:', response);
        alert('创建会话失败: ' + (response.message || '未知错误'));
      }
    } catch (error: any) {
      console.error('创建会话失败，错误详情:', error);
      console.error('错误类型:', error.constructor.name);
      console.error('错误消息:', error.message);
      console.error('错误响应:', error.response?.data);
      console.error('错误状态码:', error.response?.status);
      
      let errorMessage = '创建会话失败，请检查：\n';
      if (error.code === 'ERR_NETWORK' || error.message.includes('Network Error')) {
        errorMessage += '- 后端服务可能未启动（检查 http://localhost:8080）\n';
        errorMessage += '- 网络连接问题\n';
      } else if (error.response) {
        errorMessage += `- 服务器错误: ${error.response.status} ${error.response.statusText}\n`;
        errorMessage += `- ${JSON.stringify(error.response.data)}`;
      } else {
        errorMessage += `- ${error.message}`;
      }
      
      alert(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      {/* Hero Section */}
      <div className="max-w-4xl w-full text-center mb-12 animate-fade-in">
        <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm text-primary">AI驱动的3D生成</span>
        </div>

        <h1 className="text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
          Hunyuan3D Agent
        </h1>

        <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto">
          只需一句话，从提示词到完整的3D模型<br />
          智能优化、自动生成、实时渲染
        </p>

        {/* 输入表单 */}
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
            
            <div className="relative glass rounded-2xl p-2">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="描述你想要生成的3D模型...例如：一只可爱的猫咪坐在椅子上"
                className="w-full h-32 px-6 py-4 bg-transparent border-0 focus:outline-none text-white placeholder-gray-500 resize-none"
                disabled={isLoading}
              />

              <div className="flex items-center justify-between px-4 py-2">
                <div className="text-sm text-gray-500">
                  {prompt.length > 0 && `${prompt.length} 字符`}
                </div>

                <button
                  type="submit"
                  disabled={!prompt.trim() || isLoading}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      创建中...
                    </>
                  ) : (
                    <>
                      <Box className="w-5 h-5" />
                      开始生成
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </form>

        {/* 示例提示词 */}
        <div className="mt-8 text-left max-w-3xl mx-auto">
          <p className="text-sm text-gray-500 mb-3">试试这些示例：</p>
          <div className="flex flex-wrap gap-2">
            {[
              '一只可爱的猫咪坐在椅子上',
              '一个精美的中国瓷器花瓶',
              '一辆未来感的科幻飞车',
              '一座现代风格的双层别墅'
            ].map((example) => (
              <button
                key={example}
                onClick={() => setPrompt(example)}
                className="px-4 py-2 text-sm bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-lg transition-colors"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center text-gray-500 text-sm">
        <p>Powered by Hunyuan3D-2.1 & Gemini</p>
      </div>
    </div>
  );
}

