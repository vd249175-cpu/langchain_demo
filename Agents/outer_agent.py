from pydantic import Field
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState
from langchain.chat_models import init_chat_model
from Middleware.custom_middleware import CustomMiddleware
from Tool.nested_agent_tool import NestedAgentToolWrapper
from Tool.memory_tool import MemoryToolWrapper

class OuterAgentWrapper:
    """外层 Agent 包装类，负责静态绑定中间件拓扑并组装 OuterAgent 图。"""
    name = "outer_agent"
    system_prompt = (
        "你是一个外层主管代理，负责处理用户的文本切分任务。 "
        "你必须调用 run_inner_agent_tool 工具，将分割请求下发给内层 Agent 协调完成。 "
        "你绝对不要自己动手对文本做分割，只要做任务的分发即可。"
    )
    
    class Settings(CustomMiddleware.Settings):
        """外层 Agent 配置参数，继承外层中间件参数并追加模型运行参数。"""
        model_name: str = Field(default="qwen3.5:2b", description="外层 Agent 调用的模型名称。")
        model_provider: str = Field(default="ollama", description="模型供应商，如 'ollama', 'openai', 'anthropic' 等。")
        temperature: float = Field(default=0.0, description="外层 Agent 模型的随机温度。")
        max_tokens: int = Field(default=500, description="外层 Agent 单轮回复的最大输出长度。")

    class SubState(CustomMiddleware.SubState):
        """外层 Agent 状态，直接合并并继承所有下级组件 of SubState。"""
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
        
        # 2. 直接实例化下级中间件（其中包含其所属工具 of 级联创建）
        middlewares = [CustomMiddleware(settings=current_settings)]
        
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
