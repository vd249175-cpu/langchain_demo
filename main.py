import os
import sys
from langchain_core.messages import HumanMessage
from Agents.outer_agent import OuterAgentWrapper

def main():
    print("=== LangChain 层次闭包与参数穿透演示项目 ===")
    
    # 1. 检验本地 Ollama 服务的配置警示
    if not os.getenv("OLLAMA_MODEL") and not os.getenv("OLLAMA_BASE_URL"):
        print("提示：系统将使用默认配置连接本地 Ollama 服务")
        print("  若需指定其他地址，请在终端执行：export OLLAMA_BASE_URL=http://localhost:11434")
        print("-" * 50)
        
    # 2. 构造 Settings（通过 Wrapper 内嵌类，向上合并和向下覆盖各层组件的可调参数）
    settings = OuterAgentWrapper.Settings(
        uppercase=True,       # 这里的修改会穿透到最底层的 InnerTool，让分割后的字符变大写
        max_tool_runs=3,      # 外层中间件的工具限制门限
        custom_system_prompt="[演示自定义注入] 当用户需要切分文本时，必须立即且只能调用 run_inner_agent_tool！",
        temperature=0.0
    )
    
    print(f"正在以如下设置初始化外层 Agent (OuterAgent):")
    print(f"  模型供应商: {settings.model_provider}")
    print(f"  调用的模型: {settings.model_name}")
    print(f"  中间件单轮工具拦截上限: {settings.max_tool_runs} 次")
    print(f"  参数是否透传并大写转换: {settings.uppercase}")
    print(f"  自定义系统提示词片段: {settings.custom_system_prompt}")
    print("-" * 50)
    
    # 3. 实例化 OuterAgent 包装类，开始按拓扑装配整个树
    wrapper = OuterAgentWrapper(settings=settings)
    agent = wrapper.agent
    
    # 4. 执行任务，流式消费 trace 并打印各层的变化过程
    task = "请将文本 '双人结对编程' 分割为 3 段。"
    print(f"发起任务请求: {task}\n")
    
    try:
        # 使用 astream/stream 流式模式，传入 context=settings 将当前设置传递为运行时 context
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=task)]},
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
                        runs = data.get('completedTaskCount')
                        runs_str = f" (已累计调用: {runs} 次)" if runs is not None else ""
                        print(f"🔧 [工具调用] 开始执行 {tool_name}{runs_str}")
                    elif event == "success":
                        print(f"🔧 [工具调用] {tool_name} 执行成功！内层返回分割文本: \n{data.get('result')}")
                    else:
                        print(f"🔧 [工具调用] {tool_name} -> 动作: {event}")
                elif event_type == "nested_agent":
                    stream_mode = data.get("stream")
                    if stream_mode == "updates":
                        nodes_summary = data.get("nodes", {})
                        for node_name, summary in nodes_summary.items():
                            if isinstance(summary, dict):
                                sum_type = summary.get("type")
                                if sum_type == "tool_call":
                                    for tc in summary.get("tool_calls", []):
                                        print(f"   🤖 [内层 Agent - 模型决定调用工具] 动作: 调用 '{tc['name']}' (参数: {tc['args']})")
                                elif sum_type == "tool_response":
                                    print(f"   🤖 [内层 Agent - 工具执行返回] 内容: '{summary.get('content')}'")
                                elif sum_type == "text":
                                    print(f"   🤖 [内层 Agent - 模型直接回复] 内容: '{summary.get('content')}'")
                            else:
                                print(f"   🤖 [内层 Agent 状态] 活跃节点更新: {node_name} -> {summary}")
                    elif stream_mode == "custom":
                        custom_data = data.get("data", {})
                        inner_type = custom_data.get("type")
                        if inner_type == "tool":
                            inner_tool_name = custom_data.get("tool")
                            inner_event = custom_data.get("event")
                            if inner_event == "start":
                                print(f"      🔧 [内层工具调用] 开始执行 {inner_tool_name} (请求参数: splitCount={custom_data.get('splitCount')})")
                            elif inner_event == "success":
                                print(f"      🔧 [内层工具调用] {inner_tool_name} 成功！分割为 {custom_data.get('segmentCount')} 个段。")
                        elif inner_type == "middleware":
                            inner_stage = custom_data.get("stage")
                            inner_injected = custom_data.get("injectedPrompt")
                            inner_injected_str = f" (注入系统提示词: '{inner_injected}')" if inner_injected else ""
                            print(f"      🔄 [内层中间件钩子] {custom_data.get('middleware')} -> 阶段: {inner_stage}{inner_injected_str}")
            
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
                elif "tools" in data:
                    tools_data = data["tools"]
                    messages = tools_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        print(f"📈 [外层 Agent - 工具执行返回] 内容: '{last_msg.content}'")
                else:
                    print(f"📈 [外层 Agent 状态更新] 活跃节点: {list(data.keys())}")
                
        print("\n层次闭包演示任务圆满执行完成！")
    except Exception as e:
        print(f"\n执行演示遇到错误: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
