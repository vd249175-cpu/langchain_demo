from typing import Any, Sequence
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from Tool.inner_tool import InnerToolWrapper
from Server.demo_server import set_named_system_message

class InnerMiddleware(AgentMiddleware):
    """标准的内层代理中间件，负责挂载、实例化具体的文本分割 InnerTool，并包含自身的配置与状态。"""
    name = "inner_middleware"
    
    class Settings(InnerToolWrapper.Settings):
        """内层中间件配置，直接继承内层基础工具的 Settings。"""
        pass

    class SubState(AgentState, InnerToolWrapper.SubState, total=False):
        """内层中间件自身运行期需要记录的状态。"""
        innerMiddlewareStats: dict[str, int]
        
    state_schema = SubState

    # 直接在类级别声明覆盖挂载的工具，与 state_schema 的覆盖方式完全一致
    tools = [InnerToolWrapper().tool]

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or self.Settings()

    def before_agent(self, state: SubState, runtime: Runtime) -> dict[str, Any] | None:
        """在内层 Agent 图启动时执行，向大模型注入该中间件旗下工具的引导性系统提示词（具名插槽）。"""
        writer = runtime.stream_writer
        current_settings = runtime.context or self.settings
        
        # 1. 动态拼装工具相关的引导文本（感知 uppercase 配置）
        uppercase_text = "并且在分割完成后将英文字符自动转换为大写形式" if getattr(current_settings, "uppercase", False) else "保持英文字符的原样大小写格式"
        instruction = (
            f"你目前处于内层文本处理中间件控制下。对于输入的文本，"
            f"你必须使用 'split_text_segments' 工具进行处理，{uppercase_text}。"
        )
        
        # 2. 将命名系统消息注入到 messages 列表中，槽位名设置为 'middleware.inner_tool.guidance'
        messages = state.get("messages", [])
        messages = set_named_system_message(messages, name="middleware.inner_tool.guidance", text=instruction)
        
        if writer:
            writer({
                "type": "middleware",
                "middleware": self.name,
                "stage": "before_agent",
                "injectedPrompt": instruction
            })
        return {
            "messages": messages
        }
