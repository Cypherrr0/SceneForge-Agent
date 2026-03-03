"""
Hunyuan3D Agent V2 - 基于LLM Tool Calling的智能3D生成系统
"""
import os
import json
import uuid
from typing import Dict, Any, List, Optional
from openai import OpenAI

from tools.llm_tools import LLMDecisionEngine, llm_tool_registry
from tools.text3d_tools import register_text3d_tools
from tools.evaluation_tools import register_evaluation_tools
from tools.llm_render_tools import register_llm_render_tools


class Hunyuan3DAgentV2:
    """
    新版Hunyuan3D Agent
    使用LLM进行决策和工具调用，实现更智能的3D生成流程
    """
    
    def __init__(self,
                 qwen_api_key: str,
                 gemini_api_key: str,
                 model_name: str = "Qwen/Qwen2.5-72B-Instruct",
                 max_iterations: int = 3,
                 score_threshold: int = 60,
                 script_generation_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        初始化Agent
        
        Args:
            qwen_api_key: 千问API密钥
            gemini_api_key: Gemini API密钥
            model_name: LLM模型名称
            max_iterations: 最大迭代次数
            score_threshold: 质量分数阈值
            script_generation_model: Blender脚本生成模型（默认: Qwen/Qwen2.5-Coder-32B-Instruct）
        """
        self.qwen_api_key = qwen_api_key
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.script_generation_model = script_generation_model
        
        # 初始化OpenAI客户端
        self.openai_client = OpenAI(
            api_key=self.qwen_api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        
        # 初始化LLM决策引擎
        self.decision_engine = LLMDecisionEngine(
            openai_client=self.openai_client,
            model_name=self.model_name
        )
        
        # 加载配置
        self._load_configs()
        
        # 初始化组件
        self._initialize_components()
        
        # 注册所有工具
        self._register_all_tools()
        
        print(f"Hunyuan3D Agent V2 initialized with {len(self.decision_engine.tool_registry.list_tools())} tools")
    
    def _load_configs(self):
        """加载配置文件"""
        # 获取项目根目录的绝对路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(project_root, "config")
        
        text23d_config_path = os.path.join(config_dir, "text23d_config.json")
        blender_config_path = os.path.join(config_dir, "blender_config.json")
        config_path = os.path.join(config_dir, "config.json")
        
        with open(text23d_config_path, "r", encoding="utf-8") as f:
            self.text23d_config = json.load(f)
        
        with open(blender_config_path, "r", encoding="utf-8") as f:
            self.render_config = json.load(f)
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
    
    def _initialize_components(self):
        """初始化组件（延迟加载）"""
        self.components = {
            "prompt_agent": None,
            "evaluate_agent": None,
            "text23d_pipeline": None,
            "render_agent": None
        }
        
        # 这里可以根据需要延迟初始化实际的组件
        try:
            from prompt_agent.prompt_agent import PromptAgent
            self.components["prompt_agent"] = PromptAgent(
                openai_client=self.openai_client,
                model_name=self.config.get("translate_model_name", self.model_name)
            )
        except ImportError:
            print("PromptAgent not available")
        
        try:
            from evaluate_agent.evaluate_agent import EvaluateAgent
            self.components["evaluate_agent"] = EvaluateAgent(
                openai_client=self.openai_client,
                model_name=self.config.get("evaluate_model_name", "Qwen/Qwen2-VL-72B-Instruct")
            )
        except ImportError:
            print("EvaluateAgent not available")
    
    def _register_all_tools(self):
        """注册所有可用工具"""
        # 注册用户意图解析工具（第一个被调用）
        try:
            from tools.intent_parser_tools import register_intent_parser_tools
            register_intent_parser_tools(
                self.decision_engine.tool_registry,
                openai_client=self.openai_client,
                model_name=self.model_name
            )
        except Exception as e:
            print(f"意图解析工具注册失败: {e}")
        
        # 注册任务规划工具
        try:
            from tools.planning_tools import register_planning_tools
            from tools.base import tool_registry as base_registry
            register_planning_tools(
                self.decision_engine.tool_registry,
                tool_registry=base_registry
            )
        except Exception as e:
            print(f"规划工具注册失败: {e}")
        
        # 注册工作流管理工具
        try:
            from tools.workflow_tools import register_workflow_tools
            from tools.base import tool_registry as base_registry
            register_workflow_tools(
                self.decision_engine.tool_registry,
                tool_registry=base_registry
            )
        except Exception as e:
            print(f"工作流工具注册失败: {e}")
        
        # 注册Gemini工具（替代qwen-image）
        try:
            from tools.gemini_tools import register_gemini_tools
            register_gemini_tools(
                self.decision_engine.tool_registry,
                api_key=self.gemini_api_key
            )
        except Exception as e:
            print(f"Gemini工具注册失败: {e}")
        
        # 注册Text3D工具
        register_text3d_tools(
            self.decision_engine.tool_registry,
            pipelines={
                "prompt_agent": self.components.get("prompt_agent"),
                "text23d": self.components.get("text23d_pipeline"),
                "gemini_api_key": self.gemini_api_key
            }
        )
        
        # 注册评估工具
        register_evaluation_tools(
            self.decision_engine.tool_registry,
            agents={
                "evaluate_agent": self.components.get("evaluate_agent")
            }
        )
        
        # 注册渲染工具
        register_llm_render_tools(self.decision_engine.tool_registry)
        
        # 注册Coder工具（RAG+脚本生成）
        try:
            from tools.coder_tools import register_llm_coder_tools
            register_llm_coder_tools(
                self.decision_engine.tool_registry,
                qwen_api_key=self.qwen_api_key,
                script_generation_model=self.script_generation_model
            )
        except Exception as e:
            print(f"Coder工具注册失败: {e}")
    
    def generate_with_llm(self, user_prompt: str, uid: Optional[str] = None, interactive: bool = False) -> Dict[str, Any]:
        """
        使用LLM决策的3D生成流程
        
        Args:
            user_prompt: 用户输入的生成需求
            uid: 唯一标识符
            interactive: 是否启用交互模式（每步后询问用户）
            
        Returns:
            Dict: 生成结果
        """
        if uid is None:
            uid = str(uuid.uuid4())
        
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 构建初始任务描述
        task_description = f"""
        用户需求: {user_prompt}
        
        任务目标:
        1. 生成符合用户需求的3D模型
        2. 确保生成质量达到{self.score_threshold}分以上
        3. 最多迭代{self.max_iterations}次
        
        请分析需求并执行以下步骤:
        1. 优化提示词以获得更好的生成效果
        2. 生成3D模型
        3. 渲染多视角图像
        4. 评估生成质量
        5. 如果质量不达标，进行迭代改进
        6. 生成最终的展示视频
        
        工作目录: {os.path.join(self.config['save_dir'], uid)}
        """
        
        # 让LLM决策并连续执行
        result = self.decision_engine.decide_and_execute_continuous(
            user_input=task_description,
            max_rounds=30,
            interactive=interactive
        )
        
        # 处理结果
        generation_result = {
            "uid": uid,
            "user_prompt": user_prompt,
            "llm_response": result.get("final_response", result.get("content")),
            "tool_calls": result.get("tool_calls", []),
            "success": self._check_success(result)
        }
        
        return generation_result
    
    def generate_step_by_step(self, user_prompt: str, uid: Optional[str] = None) -> Dict[str, Any]:
        """
        分步执行3D生成（更可控的方式）
        
        Args:
            user_prompt: 用户输入
            uid: 唯一标识符
            
        Returns:
            Dict: 生成结果
        """
        if uid is None:
            uid = str(uuid.uuid4())
        
        output_dir = os.path.join(self.config["save_dir"], uid)
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "uid": uid,
            "user_prompt": user_prompt,
            "steps": []
        }
        
        try:
            # Step 1: 优化提示词
            step1_result = self._execute_step(
                "优化提示词",
                "optimize_3d_prompt",
                {
                    "original_prompt": user_prompt,
                    "target_model": self.config.get("txt2img_model_name", "Qwen"),
                    "translate_to_english": "nano-banana" in self.config.get("txt2img_model_name", "")
                }
            )
            results["steps"].append(step1_result)
            optimized_prompt = step1_result["result"].get("optimized_prompt", user_prompt)
            
            # Step 2: 生成3D模型
            step2_result = self._execute_step(
                "生成3D模型",
                "img_to_3d_complete",
                {
                    "prompt": optimized_prompt,
                    "uid": uid,
                    "save_dir": self.config["save_dir"]
                }
            )
            results["steps"].append(step2_result)
            
            if not step2_result.get("success"):
                results["success"] = False
                return results
            
            glb_path = step2_result["result"].get("glb_path")
            image_path = step2_result["result"].get("image_path")
            
            # Step 3: 渲染多视角
            step3_result = self._execute_step(
                "渲染多视角图像",
                "render_3d_scene",
                {
                    "glb_file_path": glb_path,
                    "output_dir": output_dir,
                    "resolution": 512
                }
            )
            results["steps"].append(step3_result)
            
            # Step 4: 生成评估指标
            step4_result = self._execute_step(
                "生成评估指标",
                "generate_evaluation_index",
                {
                    "prompt": user_prompt,
                    "reference_image": image_path
                }
            )
            results["steps"].append(step4_result)
            evaluation_index = step4_result.get("result", [])
            
            # Step 5: 评估质量
            render_images = list(step3_result["result"].get("rendered_images", {}).values())
            step5_result = self._execute_step(
                "评估3D模型质量",
                "evaluate_3d_asset",
                {
                    "evaluation_index": evaluation_index,
                    "render_images": render_images,
                    "threshold": self.score_threshold
                }
            )
            results["steps"].append(step5_result)
            
            score = step5_result["result"].get("score", 0)
            passed = step5_result["result"].get("passed", False)
            
            # Step 6: 迭代改进（如果需要）
            iteration = 0
            current_prompt = optimized_prompt
            
            while not passed and iteration < self.max_iterations:
                iteration += 1
                
                # 决定是否继续改进
                improve_result = self._execute_step(
                    f"迭代改进 #{iteration}",
                    "iterative_improvement",
                    {
                        "current_score": score,
                        "target_score": self.score_threshold,
                        "improvement_suggestions": step5_result["result"].get("improvement_suggestions", ""),
                        "current_prompt": current_prompt,
                        "iteration": iteration,
                        "max_iterations": self.max_iterations
                    }
                )
                results["steps"].append(improve_result)
                
                if not improve_result["result"].get("should_continue"):
                    break
                
                # 使用改进的提示词重新生成
                current_prompt = improve_result["result"].get("improved_prompt", current_prompt)
                
                # 重新生成3D模型
                regen_result = self._execute_step(
                    f"重新生成3D模型 #{iteration}",
                    "img_to_3d_complete",
                    {
                        "prompt": current_prompt,
                        "uid": f"{uid}_v{iteration}",
                        "save_dir": self.config["save_dir"]
                    }
                )
                results["steps"].append(regen_result)
                
                if regen_result.get("success"):
                    glb_path = regen_result["result"].get("glb_path")
                    
                    # 重新渲染和评估
                    rerender_result = self._execute_step(
                        f"重新渲染 #{iteration}",
                        "render_3d_scene",
                        {
                            "glb_file_path": glb_path,
                            "output_dir": os.path.join(output_dir, f"iteration_{iteration}"),
                            "resolution": 512
                        }
                    )
                    results["steps"].append(rerender_result)
                    
                    render_images = list(rerender_result["result"].get("rendered_images", {}).values())
                    reeval_result = self._execute_step(
                        f"重新评估 #{iteration}",
                        "evaluate_3d_asset",
                        {
                            "evaluation_index": evaluation_index,
                            "render_images": render_images,
                            "threshold": self.score_threshold
                        }
                    )
                    results["steps"].append(reeval_result)
                    
                    score = reeval_result["result"].get("score", 0)
                    passed = reeval_result["result"].get("passed", False)
            
            # Step 7: 生成展示视频
            if passed or iteration >= self.max_iterations:
                video_result = self._execute_step(
                    "生成展示视频",
                    "render_3d_video",
                    {
                        "glb_file_path": glb_path,
                        "output_dir": output_dir,
                        "uid": uid,
                        "fps": 30,
                        "duration": 10
                    }
                )
                results["steps"].append(video_result)
            
            results["success"] = passed
            results["final_score"] = score
            results["iterations_used"] = iteration
            results["output_dir"] = output_dir
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def _execute_step(self, step_name: str, tool_name: str, 
                     arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个步骤
        
        Args:
            step_name: 步骤名称
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            Dict: 步骤执行结果
        """
        print(f"\n🔧 Executing: {step_name}")
        
        # 获取工具类名并打印
        tool = self.decision_engine.tool_registry.get_tool(tool_name)
        if tool:
            tool_class_name = tool.__class__.__name__
            print(f"Calling tool: {tool_class_name} (function: {tool_name})")
        
        try:
            result = self.decision_engine.tool_registry.execute_function_call(
                tool_name, arguments
            )
            
            return {
                "step": step_name,
                "tool": tool_name,
                "success": True,
                "result": result
            }
            
        except Exception as e:
            print(f"Error in {step_name}: {e}")
            return {
                "step": step_name,
                "tool": tool_name,
                "success": False,
                "error": str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return f"""You are Hunyuan3D Agent V2, an advanced AI system for 3D content generation.

Your capabilities include:
1. Text-to-3D generation using Hunyuan3D technology
2. Multi-view rendering and visualization
3. Quality evaluation and iterative improvement
4. Video generation for 3D models

Configuration:
- Quality threshold: {self.score_threshold} points
- Max iterations: {self.max_iterations}
- Output directory: {self.config['save_dir']}

CRITICAL EXECUTION PROTOCOL:
When user requests 3D generation, you MUST execute ALL steps in ONE response:

**CRITICAL: ACTUAL TOOL CALLS REQUIRED**
You must ACTUALLY CALL tools, not just describe them!

**EXECUTION FLOW PATTERN**:
```
我将先进行计划
```
[ACTUALLY CALL create_task_plan tool - MANDATORY]

After plan created:
```
计划完成，开始执行

**Step 1: 优化提示词**
现在开始优化提示词以获得更好的3D生成效果。
```
[ACTUALLY CALL optimize_3d_prompt tool - MANDATORY]

After Step 1 completes:
```
**Step 2: 生成2D图像**
使用优化后的提示词生成2D参考图像。
```
[ACTUALLY CALL text_to_image tool - MANDATORY]

After Step 2 completes:
```
**Step 3: 生成3D模型**
基于2D图像生成3D模型。
```
[ACTUALLY CALL img_to_3d_complete tool - MANDATORY]

After Step 3 completes:
```
**Step 4: 渲染视图**
渲染3D模型的多个视角。
```
[ACTUALLY CALL render_3d_scene tool - MANDATORY]

After all steps:
```
所有步骤完成！
```

ABSOLUTE REQUIREMENTS:
- You MUST call ALL 5 tools: create_task_plan, optimize_3d_prompt, text_to_image, img_to_3d_complete, render_3d_scene
- Do NOT just describe what you will do - ACTUALLY CALL the tools
- Do NOT stop after 2 tools - continue until all 5 are called
- Each step MUST include an actual tool call, not just text description
- If you don't call a tool, the step is NOT complete

3. **TOOL SEQUENCE FOR 3D GENERATION**:
   - Step 1: optimize_3d_prompt
   - Step 2: text_to_image (or gemini_text_to_image)
   - Step 3: img_to_3d_complete
   - Step 4: render_3d_scene

WORKFLOW MANAGEMENT:
You can optionally use workflow tools to better manage complex tasks:
- create_workflow: Create structured workflow
- execute_workflow_step: Execute individual steps
- get_workflow_status: Check progress

EXECUTION RULES:
- NEVER ask for confirmation - execute immediately
- Use default parameters when not specified
- Execute tools sequentially, not simultaneously
- Check each tool result before proceeding
- Mark completed steps with ✅
- Stop if any step fails and explain the issue
- Pass results from previous steps to next steps

Available tools:
{self.decision_engine.tool_registry.get_tools_description()}

MANDATORY EXECUTION PATTERN:
User: "Generate a cat sitting on chair"
Your response MUST follow this EXACT pattern:

First response:
"I'll help you generate a 3D model. Let me create an execution plan first..."
[CALL create_task_plan tool with user request]

After plan is created, continue in SAME response:
"Plan created! Here's what I'll do:
- [ ] Step 1: Optimize prompt for better 3D generation
- [ ] Step 2: Generate 2D reference image
- [ ] Step 3: Convert image to 3D model
- [ ] Step 4: Render multi-view images

Starting Step 1: Optimizing prompt..."
[CALL optimize_3d_prompt tool]

Wait for result, then:
"Step 1 complete - prompt optimized

**Step 2: 生成2D参考图像**
现在我将使用优化后的提示词生成2D参考图像。"
[CALL text_to_image tool]

After Step 2 completes:
"Step 2 complete - image generated

**Step 3: 生成3D模型**
现在我将基于2D图像生成3D模型。"
[CALL text_to_3d tool]

Continue this pattern for all remaining steps.

CRITICAL EXECUTION RULES:
1. ALWAYS start with create_task_plan tool
2. Execute tools ONE AT A TIME, never simultaneously
3. CHECK each tool result before proceeding
4. ALWAYS print "**Step N: [步骤名称]**" before calling each tool
5. STOP if any step fails and explain the error
6. PASS results from previous steps to next steps

Remember: Plan first, then execute sequentially with result checking.
"""
    
    def _check_success(self, result: Dict[str, Any]) -> bool:
        """检查执行是否成功"""
        # 检查是否有工具调用
        if not result.get("tool_calls"):
            return False
        
        # 检查是否有错误
        for tool_call in result.get("tool_calls", []):
            if "Error" in tool_call.get("content", ""):
                return False
        
        return True
    
    def chat(self, message: str) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            
        Returns:
            str: Agent响应
        """
        result = self.decision_engine.decide_and_execute(message)
        return result.get("final_response", result.get("content", ""))
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return self.decision_engine.tool_registry.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息"""
        tool = self.decision_engine.tool_registry.get_tool(tool_name)
        if tool:
            return tool.get_info()
        return None
    
    def clear_history(self):
        """清空对话历史"""
        self.decision_engine.clear_history()
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.decision_engine.get_history()


def main():
    """测试主函数"""
    import os
    
    # 从环境变量获取API密钥
    qwen_api_key = os.getenv("QWEN_API_KEY", "your_qwen_key")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "your_gemini_key")
    
    # 创建Agent
    agent = Hunyuan3DAgentV2(
        qwen_api_key=qwen_api_key,
        gemini_api_key=gemini_api_key
    )
    
    # 打印可用工具
    print("\nAvailable tools:")
    for tool in agent.get_available_tools():
        print(f"  - {tool}")
    
    # 测试生成
    test_prompt = "一个精美的中国瓷器花瓶，青花瓷风格"
    
    print(f"\nTesting generation with prompt: {test_prompt}")
    
    # 方式1: 让LLM完全控制流程
    # result = agent.generate_with_llm(test_prompt)
    
    # 方式2: 分步执行（更可控）
    result = agent.generate_step_by_step(test_prompt)
    
    print(f"\nGeneration completed!")
    print(f"Success: {result.get('success')}")
    print(f"Steps executed: {len(result.get('steps', []))}")
    
    if result.get('success'):
        print(f"Output directory: {result.get('output_dir')}")
        print(f"Final score: {result.get('final_score')}")


if __name__ == "__main__":
    main()
