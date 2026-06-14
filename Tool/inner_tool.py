from pydantic import BaseModel, Field
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.agents.middleware import AgentState
from Server.demo_server import update_nested_count

def split_text_logic(text: str, split_count: int, uppercase: bool = False) -> list[str]:
    """文本分割的纯业务逻辑 helper 函数。"""
    content = text.strip()
    if not content:
        return []
    if uppercase:
        content = content.upper()
    segment_count = min(split_count, len(content))
    segments = []
    for index in range(segment_count):
        start = round(index * len(content) / segment_count)
        end = round((index + 1) * len(content) / segment_count)
        segments.append(content[start:end])
    return segments

class InnerToolWrapper:
    """内层工具包装类，负责静态挂载配置、状态声明并构建标准 @tool。"""
    name = "split_text_segments"
    description = "将输入的文本分割为指定数量的片段。"

    class Settings(BaseModel):
        """内层工具的可配置参数。"""
        uppercase: bool = Field(default=False, description="是否将输出的文本切片自动转换为大写。")
        split_text_templates: dict[str, str] = Field(
            default={
                "success": "成功将文本分割为 {count} 段：\n{segments}",
                "error": "分割文本失败：{error}"
            },
            description="文本分割工具返回的文本消息模板。"
        )

    class SubState(AgentState, total=False):
        """声明内层工具自己负责写回状态字典的字段。"""
        innerToolStats: dict[str, int]

    # 声明暴露给大模型的工具参数
    class InputSchema(BaseModel):
        text: str = Field(description="需要进行分割处理 of 源文本。")
        splitCount: int = Field(description="需要分割出的片段数量。", ge=1)

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
        def split_text_segments(runtime: ToolRuntime, text: str, splitCount: int) -> Command:
            """标准 LangChain 工具入口。"""
            context = runtime.context or current_settings
            writer = runtime.stream_writer
            
            # 读取累计运行次数并加 1
            completed_runs = int(runtime.state.get("innerToolStats", {}).get("completedInputCount", 0) or 0) + 1
            
            if writer:
                writer({
                    "type": "tool",
                    "tool": self.name,
                    "event": "start",
                    "splitCount": splitCount,
                    "uppercase": context.uppercase,
                    "completedInputCount": completed_runs
                })
            
            templates = getattr(context, "split_text_templates", {})
            try:
                segments = split_text_logic(text, splitCount, uppercase=context.uppercase)
                
                # 从配置模板生成成功消息
                success_tpl = templates.get("success", "成功将文本分割为 {count} 段：\n{segments}")
                message_text = success_tpl.format(count=len(segments), segments="\n".join(segments))
                
                if writer:
                    writer({
                        "type": "tool",
                        "tool": self.name,
                        "event": "success",
                        "segmentCount": len(segments)
                    })
                
                return Command(
                    update={
                        "messages": [ToolMessage(content=message_text, tool_call_id=runtime.tool_call_id)],
                        **update_nested_count(runtime.state, "innerToolStats", "completedInputCount", completed_runs)
                    }
                )
            except Exception as exc:
                # 从配置模板生成失败消息
                error_tpl = templates.get("error", "分割文本失败：{error}")
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
                
        return split_text_segments
