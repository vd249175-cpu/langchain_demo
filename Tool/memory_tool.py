from pydantic import BaseModel, Field
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.agents.middleware import AgentState
from Server.demo_server import set_named_system_message

class MemoryToolWrapper:
    """记忆工具包装类，允许 Agent 在运行期动态修改/更新特定的具名系统提示词。"""
    name = "save_agent_memory"
    description = "将重要事实或规则保存到代理的长期记忆中，以便模型在后续的推理步骤中一直遵循该记忆。"

    class Settings(BaseModel):
        """记忆工具配置。"""
        memory_message_name: str = Field(
            default="middleware.memory.guidance",
            description="具名系统消息的 slot 名称，工具将据此查找并修改对应的系统提示词。"
        )
        save_memory_templates: dict[str, str] = Field(
            default={
                "success": "记忆保存成功！当前记忆已更新为：{fact}",
                "error": "保存记忆失败：{error}"
            },
            description="保存记忆工具返回的文本模板。"
        )

    class SubState(AgentState, total=False):
        """记忆工具声明的专属状态（这里我们主要通过修改 messages 来达到目的）。"""
        pass

    class InputSchema(BaseModel):
        fact: str = Field(description="需要让 Agent 记住的重要事实或规则信息。")

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
        def save_agent_memory(runtime: ToolRuntime, fact: str) -> Command:
            """标准 LangChain 工具，通过 runtime 拿到状态并修改具名系统消息内容。"""
            context = runtime.context or current_settings
            writer = runtime.stream_writer
            
            message_name = getattr(context, "memory_message_name", "middleware.memory.guidance")
            templates = getattr(context, "save_memory_templates", {})
            
            if writer:
                writer({
                    "type": "tool",
                    "tool": self.name,
                    "event": "start",
                    "fact": fact
                })
                
            try:
                # 1. 获取当前所有的消息历史
                messages = runtime.state.get("messages", [])
                
                # 2. 找到特定 name 的 SystemMessage 并更新其内容
                memory_text = f"你的长期记忆（当前消息标识名为 '{message_name}'）：【{fact}】"
                updated_messages = set_named_system_message(messages, name=message_name, text=memory_text)
                
                # 3. 构造成功返回信息
                success_tpl = templates.get("success", "记忆保存成功！当前记忆已更新为：{fact}")
                success_message = success_tpl.format(fact=fact)
                
                if writer:
                    writer({
                        "type": "tool",
                        "tool": self.name,
                        "event": "success",
                        "fact": fact
                    })
                    
                return Command(
                    update={
                        "messages": updated_messages + [ToolMessage(content=success_message, tool_call_id=runtime.tool_call_id)]
                    }
                )
            except Exception as exc:
                error_tpl = templates.get("error", "保存记忆失败：{error}")
                error_message = error_tpl.format(error=str(exc))
                if writer:
                    writer({
                        "type": "tool",
                        "tool": self.name,
                        "event": "error",
                        "error": error_message
                    })
                return Command(
                    update={
                        "messages": [ToolMessage(content=error_message, tool_call_id=runtime.tool_call_id)]
                    }
                )
                
        return save_agent_memory
