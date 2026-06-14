from typing import Any, Sequence
from langchain.agents.middleware import AgentMiddleware, AgentState, ExtendedModelResponse, ModelRequest
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from Tool.nested_agent_tool import NestedAgentToolWrapper
from Tool.memory_tool import MemoryToolWrapper
from Server.demo_server import get_nested_count, update_nested_count, set_named_system_message

class CustomMiddleware(AgentMiddleware):
    """标准的 LangChain 代理中间件，用于向外层 Agent 提供调用限制拦截及自定义系统提示词动态注入。"""
    name = "custom_middleware"
    
    class Settings(NestedAgentToolWrapper.Settings, MemoryToolWrapper.Settings):
        """外层中间件配置参数。"""
        max_tool_runs: int = Field(default=3, description="允许的最大工具调用次数门限。")
        custom_system_prompt: str = Field(
            default="当用户需要切分文本时，必须且只能调用 run_inner_agent_tool 进行下发。",
            description="由用户或上层自定义注入的系统提示词片段。"
        )
        custom_middleware_templates: dict[str, str] = Field(
            default={
                "rejected": "工具调用请求已被拦截并拒绝：超出了 {limit} 次的最高调用限额。"
            },
            description="中间件拦截和警告返回的文本消息模板。"
        )

    class SubState(AgentState, NestedAgentToolWrapper.SubState, MemoryToolWrapper.SubState, total=False):
        """外层中间件负责维护和读写的状态字段。"""
        middlewareStats: dict[str, int]
        
    state_schema = SubState

    # 直接在类级别声明覆盖挂载的工具，与 state_schema 的覆盖方式完全一致
    tools = [
        NestedAgentToolWrapper().tool,
        MemoryToolWrapper().tool
    ]

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or self.Settings()

    def before_agent(self, state: SubState, runtime: Runtime) -> dict[str, Any] | None:
        """在 Agent 图启动时运行，用于自动初始化并注册所有具名系统消息（包括长期记忆和协调指令）。"""
        writer = runtime.stream_writer
        messages = state.get("messages", [])
        
        # 1. 注册长期记忆插槽
        message_name = getattr(self.settings, "memory_message_name", "middleware.memory.guidance")
        has_memory = any(getattr(msg, "name", None) == message_name for msg in messages)
        if not has_memory:
            default_memory = "你目前没有任何保存的长期记忆。"
            messages = set_named_system_message(
                messages,
                name=message_name,
                text=f"你的长期记忆（当前消息标识名为 '{message_name}'）：【{default_memory}】"
            )
            
        # 2. 注册协调和限制性引导规则
        guidance = (
            f"你目前是外层协调代理。{self.settings.custom_system_prompt} "
            f"注意：最大工具调用门限为 {self.settings.max_tool_runs} 次。"
        )
        messages = set_named_system_message(messages, name="middleware.split_text.guidance", text=guidance)
        
        if writer:
            writer({
                "type": "middleware",
                "middleware": self.name,
                "stage": "before_agent",
                "injectedPrompt": guidance
            })
            
        return {"messages": messages}

    def before_model(self, state: SubState, runtime: Runtime) -> dict[str, Any] | None:
        """在调用大模型前执行，进行次数统计跟踪。"""
        writer = runtime.stream_writer
        before_count = get_nested_count(state, "middlewareStats", "beforeModelRuns") + 1
        
        if writer:
            writer({
                "type": "middleware",
                "middleware": self.name,
                "stage": "before_model",
                "beforeModelRuns": before_count
            })
        
        return update_nested_count(state, "middlewareStats", "beforeModelRuns", before_count)

    def wrap_model_call(self, request: ModelRequest, handler) -> ExtendedModelResponse:
        """包裹模型调用阶段的拦截与跟踪。"""
        writer = request.runtime.stream_writer
        if writer:
            writer({
                "type": "middleware",
                "middleware": self.name,
                "stage": "wrap_model_call"
            })
        return handler(request)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """包裹并执行工具调用的拦截与门限判断（如果超过限制，直接短路拒绝）。"""
        state = request.state
        writer = request.runtime.stream_writer
        current_settings = request.runtime.context or self.settings
        
        # 统计从 nestedAgentToolStats 记录中的嵌套 Agent 工具被调用的总次数
        tool_runs = get_nested_count(state, "nestedAgentToolStats", "completedTaskCount") + 1
        
        if writer:
            writer({
                "type": "middleware",
                "middleware": self.name,
                "stage": "wrap_tool_call",
                "toolName": request.tool_call["name"],
                "toolRuns": tool_runs,
                "maxToolRuns": current_settings.max_tool_runs
            })
        
        if tool_runs > current_settings.max_tool_runs:
            templates = getattr(current_settings, "custom_middleware_templates", {})
            rejected_tpl = templates.get("rejected", "工具调用请求已被拦截并拒绝：超出了 {limit} 次的最高调用限额。")
            error_text = rejected_tpl.format(limit=current_settings.max_tool_runs)
            
            if writer:
                writer({
                    "type": "middleware",
                    "middleware": self.name,
                    "stage": "wrap_tool_call",
                    "event": "rejected",
                    "error": error_text
                })
            return Command(
                update={
                    "messages": [ToolMessage(content=error_text, tool_call_id=request.tool_call["id"], status="error")]
                }
            )
            
        result = handler(request)
        return result
