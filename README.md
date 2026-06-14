# LangChain 层次闭包与参数穿透演示项目 (LangChain Hierarchical Closure Demo)

这是一个基于 **LangChain** 与 **LangGraph** 最新标准设计的演示项目，展示了如何通过**树状嵌套闭包（Hierarchical Closures）**与**统一类包裹器模式（Wrapper Class Pattern）**来构建高内聚、低耦合、支持参数穿透与拦截的多智能体协同系统。

---

## 1. 目录结构与职责划分 (4 Directories)

项目严格划分为 4 个核心目录，各层组件职责清晰、边界明确：

*   [Tool/](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/): **工具层**。承载所有智能体可调用的具体工具或包装后的嵌套调用逻辑。
    *   [inner_tool.py](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/inner_tool.py)：底层原子文本切分工具 `split_text_segments`。
    *   [nested_agent_tool.py](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/nested_agent_tool.py)：嵌套调用工具 `run_inner_agent_tool`，桥接内外层 Agent 的执行流。
*   [Middleware/](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/): **中间件层**。继承自 `AgentMiddleware`，在模型调用和工具调用前后注入动态行为。
    *   [custom_middleware.py](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/custom_middleware.py)：外层中间件，实现单轮工具调用次数限制、动态系统提示词插槽注入。
    *   [inner_middleware.py](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/inner_middleware.py)：内层中间件，用于流式事件跟踪与简单指标收集。
*   [Agents/](file:///Users/apexwave/Desktop/langchain_yanshi/Agents/): **智能体层**。装配和导出 CompiledStateGraph 实例。
    *   [inner_agent.py](file:///Users/apexwave/Desktop/langchain_yanshi/Agents/inner_agent.py)：内层文本切分协调 Agent。
    *   [outer_agent.py](file:///Users/apexwave/Desktop/langchain_yanshi/Agents/outer_agent.py)：外层主管分发 Agent。
*   [Server/](file:///Users/apexwave/Desktop/langchain_yanshi/Server/): **公共服务层**。存放系统消息插槽操作、跨模块的状态计数更新等纯净辅助函数。
    *   [demo_server.py](file:///Users/apexwave/Desktop/langchain_yanshi/Server/demo_server.py)：辅助工具函数，包括 `get_nested_count`、`update_nested_count` 及 `set_named_system_message`。

---

## 2. 核心架构设计与分层解析

我们的整个系统是围绕着**配置（Settings）**与**状态（State）**转的。下面我们自底向上，按照 **Tool -> Middleware -> Agent** 的层次深度剖析整个闭包体系的设计痛点与解决方案。

### 2.1 工具层 (Tool Layer) 的状态与模板去中心化

工具是整个闭包的最底层实体，它承载了直接操作数据或调用其他系统的具体逻辑。然而，在 LangChain/LangGraph 原生的开发模式下，存在多个设计痛点与架构诉求：

#### 2.1.1 痛点一：不继承 `BaseTool` 基类，使用 `@tool` 与 `StructuredTool`
*   **设计抉择**：在本项目中，我们统一**不直接继承** LangChain 的 `BaseTool` 基类。
*   **原因与背景**：
    1.  `BaseTool` 功能不完整且使用繁琐。如果直接继承 `BaseTool`，我们需要手动重写 `_run` 与 `_arun` 方法，并且在处理动态的 `ToolRuntime` 注入、异步控制和上下文合并时非常不灵活。
    2.  相反，在 LangChain 中，当我们使用标准的 **`@tool`** 装饰器修饰一个本地函数时，LangChain 在底层会将其自动组装并返回一个 **`StructuredTool`** 实例（具体为 `langchain_core.tools.structured.StructuredTool`）。
    3.  `StructuredTool` 提供了极其完善 of Schema 自动提取与类型推导，并且能原生且完美地支持 `ToolRuntime` 的注入。

#### 2.1.2 痛点二：为什么选择 `ToolRuntime`？
*   **运行时上下文获取**：通过在工具函数声明中将第一个参数指定为 `runtime: ToolRuntime`，LangChain 运行时会自动注入该实例。
*   **核心作用**：`ToolRuntime` 能够帮我们获取到所有的运行时核心参数，这在传统的静态工具声明中是极其关键的：
    *   `runtime.context`：获取穿透进来的动态 `settings` 参数（例如判断是否要 `uppercase` 或读取局部的模板配置）。
    *   `runtime.state`：读取当前图的全局状态（State），便于工具了解它被调用时的历史背景。
    *   `runtime.stream_writer`：向外层流式输出自定义事件，打通多级 Agent 嵌套时的流式事件透传。
    *   `runtime.tool_call_id`：用来关联当前的工具消息 `ToolMessage`，确保大模型可以精准追踪哪条消息对应哪个工具调用。

#### 2.1.3 痛点三：工具修改 State 但无原生注册机制
*   **问题背景**：LangGraph 对运行状态（State）的校验非常严苛。如果节点/工具在返回 `Command(update={...})` 时，写入了一个图定义中**未声明**的 State Key，LangGraph 会直接忽略或抛出运行时异常。同时，使用标准 `@tool` 装饰器装饰的函数，**没有**提供任何原生的 API 来向父级 Graph 注册工具所独占或需要写入的 State 字段。
*   **解决方案 (自定义 SubState)**：
    *   在工具的包裹类（如 `InnerToolWrapper`）中，我们自定义声明了一个嵌套类 `class SubState(AgentState, total=False)`，用来显式声明该工具运行期需要写回或修改的状态字段。
    *   例如，[inner_tool.py](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/inner_tool.py) 中的 `InnerToolWrapper` 声明了其写回状态 `innerToolStats: dict[str, int]`，这为上层组件收集和提取工具状态提供了清晰的契约（Contract）。
    *   上层中间件在定义自身的 `SubState` 时，通过多重继承合并这些工具的 `SubState`，并通过 `state_schema = SubState` 类属性绑定给中间件，从而在智能体图编译阶段（`create_agent`）实现**自动提取并注册工具能够修改的状态字段**。

#### 2.1.4 痛点四：全局模板配置冲突与组件内聚性破坏
*   **问题背景**：如果将所有工具的错误返回和提示模板集中存放在一个全局的字典（如 `tool_response_templates`）中，会导致两大设计缺陷：
    1.  **Pydantic 继承覆盖冲突**：在上层 `Settings` 继承下层 `Settings` 的过程中，如果声明了同名的全局字典，Pydantic 在继承链中会直接使用子类声明完全覆盖父类声明，导致丢失其他层级的模板。
    2.  **职责错配**：工具的成功返回、异常捕获与文案格式化理应由工具自身或其直接关联的中间件独立决定，而不是由全局配置大包大揽。
*   **解决方案 (模板去中心化配置)**：
    *   我们将提示词文案进行了彻底的**分层本地化（Decentralization）**。每个工具在各自的 `Settings` 中独立声明和管理自身的返回文本模板。
    *   [inner_tool.py](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/inner_tool.py) 拥有专属的 `split_text_templates`。
    *   [nested_agent_tool.py](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/nested_agent_tool.py) 拥有专属的 `run_inner_agent_templates`。
    *   [custom_middleware.py](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/custom_middleware.py) 拥有专属的 `custom_middleware_templates`。
    *   运行时 context 穿透传递到底层时，底层组件只获取其命名的专属模板，从源头上消除了覆盖冲突，保持了工具的强自闭环特性。

#### 2.1.5 嵌套声明 `Settings` 和 `SubState` 的便利性
*   在包裹器类（Wrapper）内部，我们将 `class Settings` 和 `class SubState` 声明为**嵌套类（Nested Classes）**。
*   **为什么这样做**：这种模式在需要进行更上层的包裹或装配时，提供了极大的便利。上层的包裹类（如 Middleware 或 OuterAgent）可以直接通过 `InnerToolWrapper.Settings` 或 `InnerToolWrapper.SubState` 拿到这些声明，用于**直接继承或拼装**，从而自动完成了整个链条上配置参数 and 状态模式的向下穿透 and 向上汇聚，避免了在不同层级重复编写 schema and 字段定义的繁琐工作。

#### 2.1.6 静态属性 `name` 的重要性
*   我们在工具和智能体包装器类中都声明了静态属性 `name`（例如 `name = "split_text_segments"`）。
*   **为什么这样做**：在 LangChain / LangGraph 框架中，`name` 是定位组件和追踪消息的唯一关键标识：
    1.  **定位关键工具**：在中间件（Middleware）对工具调用进行限流、拦截（如 `wrap_tool_call`）或进行指标统计时，我们必须通过 `request.tool_call["name"]` 匹配 `name` 属性来判断当前是大模型在调用哪一个具体的工具。
    2.  **编辑或拦截关键消息**：在处理对话历史、调试日志或在 `before_model` 等钩子中动态增删/改写 SystemMessage、ToolMessage 时，我们通常需要寻找并解析 `message.name` 或 `message.tool_call_id`。如果没有明确、固定的 `name` 属性定义，系统的消息拦截、日志过滤和状态统计将变得无从下手。

---

### 2.2 中间件层 (Middleware Layer) 的设计内涵与核心机制

中间件继承自 `AgentMiddleware`，扮演着“承上启下”的网关角色。一个中间件的挂载，在工程上可以被看作是**一种独立能力的挂载**（例如：记忆能力、调用限流能力、终端持久化会话能力等）。使用中间件不仅便于功能扩展，更有利于**状态的彻底分离与解耦**。

#### 2.2.1 核心要素：`Settings`、`SubState`、`tools` 与 `state_schema` 的编译期注册
*   **状态与工具的隔离解耦**：每个中间件内部只定义和维护它自己所关心的 `SubState` 状态与 `tools` 工具列表，彻底与其他中间件或 Agent 核心的状态和逻辑解耦。
*   **`state_schema` 与 `tools` 自动注册**：
    *   在 LangChain / LangGraph 框架中，中间件类可以通过在类级别声明 `state_schema = SubState` 与 `tools = [...]` 类级别属性，来指示框架在 Agent 编译阶段（`create_agent`）**自动检测并增量注册**该中间件及旗下工具所需的全部状态字段与工具实例。
    *   这样，底层的 Tool 虽然由于装饰器限制无法自行在 Agent 中注册状态字段，但是能够借助其宿主中间件的 `state_schema` 和 `tools` 声明式类属性，轻松完成状态与工具的自动发现、注册与合并。

#### 2.2.2 系统提示词（SystemMessage）的动态注入与命名插槽（Named Slots）
*   **痛点背景**：单独的 Tool 无法直接拦截 LLM 调用，更无法在模型执行前向其上下文注入或更新系统提示词（System Prompt）。中间件则是天然的切面，能够在模型启动前对输入消息队列进行过滤与修改。
*   **双层系统提示词自动注入演示**：为了全方位展示系统提示词在多级中间件中的管理，本项目在内外层中间件中都实现了精妙的提示词动态注入：
    1.  **外层中间件 [custom_middleware.py](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/custom_middleware.py)**：
        *   在 `before_agent` 阶段，注册并注入一个初始化的长期记忆插槽 `middleware.memory.guidance`。
        *   在 `before_model` 阶段，动态注入运行限制规则插槽 `middleware.split_text.guidance`。
    2.  **内层中间件 [inner_middleware.py](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/inner_middleware.py)**：
        *   在 `before_model` 阶段，注入专属于底层文本分割工具的规则指导插槽 `middleware.inner_tool.guidance`。它能智能捕获当前的 runtime settings（如 `uppercase` 配置），将其融入到注入给大模型的系统提示词中，指挥内层模型以正确的策略调度工具。
*   **命名提示词（Named System Messages）与自我管理记忆的核心玩法**：
    *   我们在中间件注入系统提示词时，会赋予该 `SystemMessage` 一个唯一的 `name` 属性（如 `middleware.memory.guidance`）。
    *   通过赋予系统提示词特定的 `name`（相当于暴露了一个具名插槽），该中间件旗下的记忆/设置工具（Tool）便能够在运行时通过遍历消息历史，精确找到对应 `name` 的 `SystemMessage` 并直接修改其内容。
    *   例如：当 Agent 决定“记住”某件事时，它调用 `save_memory` 工具，该工具通过 `name` 定位到那个具名系统提示词并更新其文本。下一次模型调用前，更新后的系统提示词会被大模型读取，从而实现 **Agent 自己管理自己的运行时记忆**，实现了 Tool 与 Middleware 的深度协同。

#### 2.2.3 规范设计：在中间件类声明中直接覆盖定义 `tools`
*   **类级覆盖规范**：为了符合最简洁的声明式设计，我们将工具的挂载直接放在中间件的类声明中进行属性覆盖（就如同声明 `state_schema = SubState` 一样）。我们不再在 `__init__` 中使用 `self.tools`，也不再在 Agent 构建期的 `create_agent` 函数中显式传递 `tools` 参数：
    *   例如：在 [custom_middleware.py](file:///Users/apexwave/Desktop/langchain_yanshi/Middleware/custom_middleware.py) 类定义中直接覆盖：
        ```python
        class CustomMiddleware(AgentMiddleware):
            name = "custom_middleware"
            state_schema = SubState

            # 声明式直接挂载工具列表，与 state_schema 写法高度一致
            tools = [
                NestedAgentToolWrapper().tool,
                MemoryToolWrapper().tool
            ]
        ```
    *   **内聚与合并**：这样，当我们在 Agent 构建器（如 [outer_agent.py](file:///Users/apexwave/Desktop/langchain_yanshi/Agents/outer_agent.py)）中装配中间件 `middlewares = [CustomMiddleware(settings=current_settings)]` 时，框架在编译期会自动检测中间件类并自动提取和挂载 `tools` 列表，实现了干净、优雅的声明式架构。同时，工具在运行时可以通过 `runtime.context` 动态且安全地获取穿透下来的 `settings` 配置，完全无需在类级别硬编码配置参数。

#### 2.2.4 高级用法：长生命周期与持续会话维持（如持久化终端 Terminal）
*   **单纯的 Tool 无法做到**：普通的 Tool 是无状态的、短路式的函数调用，在完成单次执行后即被销毁，它无法在多轮对话中维持一个跨交互周期的持续资源（如：一个打开的交互式 Bash Shell 终端、一个持续连接的 SSH 通道或一个持久数据库连接）。
*   **中间件的维系能力**：
    *   中间件存在于 Agent 的整个运行生命周期中，能够拦截模型的每一次输入输出和工具的每一次调用。因此，我们可以在中间件中实现并托管像**持久终端（Persistent Terminal）**这类的长生命周期资源。
    *   具体实现方案可参考官方 **DeepAgent 中的 Shell 中间件**：它通过中间件来初始化并维持底层的持久 PTY/Shell 终端进程。大模型通过调用配套的命令输入工具向该终端发送指令，而中间件则持续监听并拦截输出，将终端的实时状态与持续缓冲区内容同步注入到模型的上下文中，这在纯 Tool 架构下是绝对无法实现的。

#### 2.2.5 官方自带钩子方法的固定写法与参数传递
`AgentMiddleware` 提供了多个生命周期切面钩子，它们的参数传递与返回格式具有固定的语法契约：

1.  **`before_agent(self, state: State, runtime: Runtime) -> dict[str, Any] | None`**
    *   **执行时机**：在 Agent 图启动（即进入第一个 Node 前）时被最先触发。
    *   **作用**：接收全局的 `state` 与运行时 `runtime` 引用。如果需要修改 Graph 状态，可在此返回更新的字典；若不作修改则返回 `None`。
2.  **`before_model(self, state: State, runtime: Runtime) -> dict[str, Any] | None`**
    *   **执行时机**：在每次调用大模型前被触发（大模型可能因思考或反思被循环调用多次）。
    *   **作用**：最适合用来动态插入、修改或精简 `SystemMessage` 的内容。参数与返回值格式与 `before_agent` 相同。
3.  **`wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse`**
    *   **执行时机**：包裹大模型的物理请求阶段。
    *   **参数契约**：`request` 携带了本次模型请求的所有参数（如 messages、tools 等），而 `handler` 是底层的真实调用句柄。
    *   **作用**：我们必须在方法内调用并返回 `handler(request)`，可以在调用前后进行耗时监控、请求防刷检查、模型输入参数过滤，或者通过重写返回值直接截断模型响应。
4.  **`wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command`**
    *   **执行时机**：包裹工具执行节点的阶段。
    *   **参数契约**：`request` 包含了本次被调用的工具名称、参数 ID 等。`handler` 为实际工具执行句柄。
    *   **作用**：用于进行工具调用限流（如本项目中 `max_tool_runs` 的限制拦截）。如果判定超出限额，中间件可以直接拦截而不执行 `handler(request)`，转而直接构造并返回一个包含错误提示的 `Command` 对象，安全地将执行流短路返回给模型。

---

### 2.3 智能体层 (Agent Layer) 的闭包与参数穿透

智能体层是配置和运行流的最终汇聚点。

*   **模型与供应商的自由配置**：
    *   彻底去除了陈旧固化的 Model 目录。大模型的供应商 (`model_provider`) 和具体型号 (`model_name`) 被全部参数化为 `Settings` 中的属性（默认配置为 `ollama` 供应商及 `qwen3.5:2b` 模型）。
    *   在智能体初始化时，通过标准的 `init_chat_model` 实现动态组装，将模型选型与厂商的决定权彻底交还给应用开发者。
*   **统一的参数穿透链 (Settings Chain)**：
    *   `Settings` 统一使用 Pydantic BaseModel 声明，重命名避免与 LangChain 原生 `Config` 冲突。
    *   外层 `OuterAgentWrapper.Settings` 继承自 `CustomMiddleware.Settings`，后者继承自底层嵌套工具 `NestedAgentToolWrapper.Settings`。
    *   这形成了一条完美的配置链条：在 `main.py` 实例化外层 Agent 时，我们只需要传入一个外层 `settings` 实例。这个实例中关于底层原子工具（如 `uppercase` 大写控制）的参数设定，不需要任何手动的打包和提取，就能够通过 `context=settings` 自动穿透传递到最底层的 [inner_tool.py](file:///Users/apexwave/Desktop/langchain_yanshi/Tool/inner_tool.py) 逻辑中。

---

## 3. 命名与规范约定 (Naming Conventions)

为维持系统极高的内聚性和强自描述性，我们约定了统一的包裹器骨架：

1.  **Wrapper 包裹类**：组件的外层自描述容器，命名为 `XxxWrapper`（如 `InnerAgentWrapper`、`CustomMiddleware`）。
2.  **配置定义**：统一在包裹类中定义嵌套类 `class Settings(BaseModel)`。
3.  **状态定义**：统一在包裹类中定义嵌套类 `class SubState(AgentState, total=False)`。
4.  **智能体系统提示词**：类级别静态属性 `system_prompt = (...)`。
5.  **中间件状态注册**：子类类属性 `state_schema = SubState`。
6.  **工具输入参数声明**：在工具类内部声明 `class InputSchema(BaseModel)`，绑定为 `@tool(args_schema=InputSchema)`。

---

## 4. 演示与调试指南 (Running & Debugging)

### 4.1 快速运行演示
确保本地环境已安装 `uv` 并且已启动本地 Ollama 服务（已拉取 `qwen3.5:2b` 模型）。

在终端中执行：
```bash
# 启动演示项目
uv run python main.py
```

### 4.2 查看所有可配置参数
执行参数反射脚本，可直接拉出包含说明和默认值在内的完整配置 Schema 列表：
```bash
uv run python print_settings.py
```
