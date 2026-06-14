from typing import Any
from pydantic import BaseModel, Field
from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from Agents.inner_agent import InnerAgentWrapper
from Server.demo_server import update_nested_count, get_nested_count

def summarize_inner_chunk(chunk: Any) -> dict[str, Any]:
    """压缩和格式化内层 Agent 的 stream 事件块，便于外层 tool 实时转发。"""
    if isinstance(chunk, tuple) and len(chunk) == 2:
        stream_type, payload = chunk
    else:
        stream_type = getattr(chunk, "type", None) or (chunk.get("type") if isinstance(chunk, dict) else "unknown")
        payload = chunk.get("data", chunk) if isinstance(chunk, dict) else chunk
        
    if stream_type == "custom":
        return {"stream": "custom", "data": payload}
    elif stream_type == "updates" and isinstance(payload, dict):
        summary = {}
        for node_name, update in payload.items():
            if isinstance(update, dict) and "messages" in update:
                messages = update["messages"]
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        summary[node_name] = {
                            "type": "tool_call",
                            "tool_calls": [
                                {
                                    "name": tc["name"],
                                    "args": tc["args"],
                                    "id": tc.get("id")
                                }
                                for tc in last_msg.tool_calls
                            ]
                        }
                    elif hasattr(last_msg, "content"):
                        msg_type = "tool_response" if last_msg.__class__.__name__ == "ToolMessage" else "text"
                        summary[node_name] = {
                            "type": msg_type,
                            "content": last_msg.content,
                            "name": getattr(last_msg, "name", None),
                            "tool_call_id": getattr(last_msg, "tool_call_id", None)
                        }
            else:
                summary[node_name] = sorted(update.keys()) if isinstance(update, dict) else str(update)
        return {
            "stream": "updates",
            "nodes": summary
        }
    return {"stream": str(stream_type)}

class NestedAgentToolWrapper:
    """嵌套 Agent 工具包装类，负责调用内层 Agent 并挂载配置、状态声明。"""
    name = "run_inner_agent_tool"
    description = "将复杂的文本分割任务委托给内层 Agent 协调完成。"

    class Settings(InnerAgentWrapper.Settings):
        """外层工具配置参数，直接继承内层 Agent 的 Settings。"""
        run_inner_agent_templates: dict[str, str] = Field(
            default={
                "success": "内层 Agent 执行成功，切分结果如下：\n{result}",
                "error": "嵌套调用内层 Agent 执行失败：{error}"
            },
            description="嵌套 Agent 调用工具返回的文本消息模板。"
        )

    class SubState(InnerAgentWrapper.SubState, total=False):
        """外层嵌套工具负责写回的状态。"""
        nestedAgentToolStats: dict[str, int]

    class InputSchema(BaseModel):
        """暴露给外层大模型的参数。"""
        task: str = Field(description="交付给内层 Agent 完成的文本切分任务。")

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or self.Settings()
        self.tool = self.create_tool()

    def create_tool(self):
        current_settings = self.settings
        
        @tool(
            self.name,
            args_schema=self.InputSchema,
            description=self.description
        )
        def run_inner_agent_tool(runtime: ToolRuntime, task: str) -> Command:
            """标准 LangChain 嵌套工具入口。"""
            context = runtime.context or current_settings
            writer = runtime.stream_writer
            
            completed_tasks = get_nested_count(runtime.state, "nestedAgentToolStats", "completedTaskCount") + 1
            
            if writer:
                writer({
                    "type": "tool",
                    "tool": self.name,
                    "event": "start",
                    "completedTaskCount": completed_tasks
                })
            
            try:
                # 1. 实例化内层 Agent 包装类，并继续向下传当前的 settings context
                inner_wrapper = InnerAgentWrapper(settings=context)
                inner_agent = inner_wrapper.agent
                
                # 2. 流式（Stream）运行内层 Agent，捕获内层 updates 与 custom 事件，重新打包向外发射
                inner_result_msg = "未获取到具体分割结果"
                for chunk in inner_agent.stream(
                    {"messages": [HumanMessage(content=task)]},
                    context=context,
                    stream_mode=["custom", "updates"],
                    version="v2"
                ):
                    rendered = summarize_inner_chunk(chunk)
                    if writer:
                        writer({
                            "type": "nested_agent",
                            "tool": self.name,
                            **rendered
                        })
                    # 捕获内层 Agent 的模型输出作为结果
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        stream_type, payload = chunk
                    else:
                        stream_type = chunk.get("type") if isinstance(chunk, dict) else "unknown"
                        payload = chunk.get("data") if isinstance(chunk, dict) else chunk

                    if stream_type == "updates" and isinstance(payload, dict) and "model" in payload:
                        model_update = payload["model"]
                        if "messages" in model_update and model_update["messages"]:
                            last_msg = model_update["messages"][-1]
                            # 如果是 AI 消息或具有 content 属性的消息
                            if hasattr(last_msg, "content") and last_msg.content:
                                inner_result_msg = last_msg.content
                            elif isinstance(last_msg, dict) and last_msg.get("content"):
                                inner_result_msg = last_msg["content"]
                
                if writer:
                    writer({
                        "type": "tool",
                        "tool": self.name,
                        "event": "success",
                        "completedTaskCount": completed_tasks,
                        "result": inner_result_msg
                    })
                
                templates = getattr(context, "run_inner_agent_templates", {})
                success_tpl = templates.get("success", "内层 Agent 执行成功，切分结果如下：\n{result}")
                success_message = success_tpl.format(result=inner_result_msg)
                
                return Command(
                    update={
                        "messages": [ToolMessage(content=success_message, tool_call_id=runtime.tool_call_id)],
                        **update_nested_count(runtime.state, "nestedAgentToolStats", "completedTaskCount", completed_tasks)
                    }
                )
            except Exception as exc:
                templates = getattr(context, "run_inner_agent_templates", {})
                error_tpl = templates.get("error", "嵌套调用内层 Agent 执行失败：{error}")
                error_text = error_tpl.format(error=str(exc))
                
                if writer:
                    writer({
                        "type": "tool",
                        "tool": self.name,
                        "event": "error",
                        "error": error_text
                    })
                return Command(
                    update={
                        "messages": [ToolMessage(content=error_text, tool_call_id=runtime.tool_call_id)]
                    }
                )
                
        return run_inner_agent_tool
