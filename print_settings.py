from Agents.outer_agent import OuterAgentWrapper

def print_all_settings():
    print("=== LangChain 层次闭包项目所有可配置参数 ===")
    
    settings_cls = OuterAgentWrapper.Settings
    # 获取 Pydantic 模型的 JSON Schema 字典
    schema = settings_cls.model_json_schema()
    
    properties = schema.get("properties", {})
    for name, prop in properties.items():
        prop_type = prop.get("type", "未知")
        default = prop.get("default", "无默认值")
        description = prop.get("description", "无参数描述说明")
        print(f"\n参数名称: {name}")
        print(f"  类型: {prop_type}")
        print(f"  默认值: {default}")
        print(f"  说明: {description}")
        print("-" * 40)

if __name__ == "__main__":
    print_all_settings()
