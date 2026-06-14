from pydantic import Field
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState
from langchain.chat_models import init_chat_model
from Middleware.inner_middleware import InnerMiddleware
from Tool.inner_tool import InnerToolWrapper

class InnerAgentWrapper:
    """内层 Agent 包装类，负责静态绑定中间件拓扑并组装 InnerAgent 图。"""
    name = "inner_agent"
    system_prompt = (
        "你是一个底层的纯文本处理助手。你的唯一任务是分割文本。 "
        "必须调用 split_text_segments 工具来处理文本分割，绝不能自己编造结果。 "
        "当工具返回结果后，直接原样将切分结果返回给用户。"
    )
    
    class Settings(InnerMiddleware.Settings):
        """内层 Agent 配置参数，继承内层中间件参数并追加模型必要运行参数。"""
        model_name: str = Field(default="qwen3.5:2b", description="内层 Agent 调用的模型名称。")
        model_provider: str = Field(default="ollama", description="模型供应商，如 'ollama', 'openai', 'anthropic' 等。")
        temperature: float = Field(default=0.0, description="内层 Agent 模型的随机温度。")
        max_tokens: int = Field(default=400, description="内层 Agent 单轮回复的最大输出长度。")

    class SubState(InnerMiddleware.SubState):
        """内层 Agent 状态，直接合并并继承所有下级组件 of SubState。"""
        pass

    def __init__(self, settings: Settings | None = None, checkpointer=None):
        self.settings = settings or self.Settings()
        self.checkpointer = checkpointer
        self.agent = self.create_agent()

    def create_agent(self):
        current_settings = self.settings
        # 1. 直接构造大模型实例，支持供应商和模型名称自定义
        model = init_chat_model(
            model=current_settings.model_name,
            model_provider=current_settings.model_provider,
            temperature=current_settings.temperature
        )
        
        # 2. 直接实例化下级中间件（其中包含其所属工具的级联创建）
        middlewares = [InnerMiddleware(settings=current_settings)]
        
        # 3. 装配 LangChain 标准 Agent 并返回
        return create_agent(
            model=model,
            system_prompt=self.system_prompt,
            middleware=middlewares,
            state_schema=self.SubState,
            context_schema=type(current_settings),
            checkpointer=self.checkpointer,
            name=self.name
        )
