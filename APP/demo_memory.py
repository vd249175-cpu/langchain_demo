import os
import sys
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Ensure the parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.outer_agent import OuterAgentWrapper

def main():
    print("=== LangChain 记忆工具演示 (Demo Memory Tool) ===")
    
    # 1. 检验本地 Ollama 服务的配置警示
    if not os.getenv("OLLAMA_MODEL") and not os.getenv("OLLAMA_BASE_URL"):
        print("提示：系统将使用默认配置连接本地 Ollama 服务")
        print("  若需指定其他地址，请在终端执行：export OLLAMA_BASE_URL=http://localhost:11434")
        print("-" * 50)
        
    # 2. 构造 Settings
    settings = OuterAgentWrapper.Settings(
        max_tool_runs=3,
        custom_system_prompt="[演示记忆注入] 作为一个智能助理，你可以调用 save_agent_memory 工具保存用户的偏好或信息。",
        temperature=0.0
    )
    
    # 3. 实例化 OuterAgent，传入 InMemorySaver 记忆检查点
    checkpointer = MemorySaver()
    wrapper = OuterAgentWrapper(settings=settings, checkpointer=checkpointer)
    agent = wrapper.agent
    
    # 配置 thread_id 以便在多轮对话中保持状态
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    
    # --- 第一轮：要求 Agent 记住一个事实 ---
    task1 = "请记住：我最喜欢的编程语言是 Python。"
    print(f"【第一轮对话】发起请求: {task1}\n")
    
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=task1)]},
            config=config,
            context=settings,
            stream_mode=["custom", "updates"],
            version="v2"
        ):
            stream_type = chunk[0] if isinstance(chunk, tuple) else chunk.get("type", "unknown")
            data = chunk[1] if isinstance(chunk, tuple) else chunk.get("data", chunk)
            
            if stream_type == "custom":
                event_type = data.get("type")
                if event_type == "middleware":
                    stage = data.get("stage")
                    middleware_name = data.get("middleware")
                    injected = data.get("injectedPrompt")
                    injected_str = f" (注入系统提示词: '{injected}')" if injected else ""
                    print(f"🔄 [外层中间件钩子] {middleware_name}拦截 -> 阶段: {stage}{injected_str}")
                elif event_type == "tool":
                    tool_name = data.get("tool")
                    event = data.get("event")
                    if event == "start":
                        print(f"🔧 [工具调用] 开始执行 {tool_name} (保存事实: '{data.get('fact')}')")
                    elif event == "success":
                        print(f"🔧 [工具调用] {tool_name} 执行成功！已存入记忆。")
            elif stream_type == "updates":
                if "model" in data:
                    model_data = data["model"]
                    messages = model_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            tool_call = last_msg.tool_calls[0]
                            print(f"📈 [外层 Agent - 模型决定调用工具] 动作: 调用 '{tool_call['name']}' (参数: {tool_call['args']})")
                        elif hasattr(last_msg, "content") and last_msg.content:
                            print(f"📈 [外层 Agent - 模型直接回复] 内容: '{last_msg.content}'")

        print("\n" + "="*60 + "\n")
        
        # --- 第二轮：问 Agent 记住了什么 ---
        task2 = "我最喜欢的编程语言是什么？"
        print(f"【第二轮对话】发起请求: {task2}\n")
        
        # 在多轮对话中，我们直接传入新消息，MemorySaver 会自动加载历史消息和更新后的 SystemMessage
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=task2)]},
            config=config,
            context=settings,
            stream_mode=["custom", "updates"],
            version="v2"
        ):
            stream_type = chunk[0] if isinstance(chunk, tuple) else chunk.get("type", "unknown")
            data = chunk[1] if isinstance(chunk, tuple) else chunk.get("data", chunk)
            
            if stream_type == "custom":
                event_type = data.get("type")
                if event_type == "middleware":
                    stage = data.get("stage")
                    middleware_name = data.get("middleware")
                    injected = data.get("injectedPrompt")
                    injected_str = f" (注入系统提示词: '{injected}')" if injected else ""
                    print(f"🔄 [外层中间件钩子] {middleware_name}拦截 -> 阶段: {stage}{injected_str}")
            elif stream_type == "updates":
                if "model" in data:
                    model_data = data["model"]
                    messages = model_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content") and last_msg.content:
                            print(f"📈 [外层 Agent - 模型直接回复] 内容: '{last_msg.content}'")

        print("\n记忆工具演示任务圆满执行完成！")
    except Exception as e:
        print(f"\n执行演示遇到错误: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
