"""共享的辅助工具函数。"""

from typing import Any
from langchain_core.messages import AnyMessage, SystemMessage



def get_nested_count(state: dict[str, Any], namespace: str, key: str) -> int:
    """安全读取 State 嵌套命名空间字典中的计数值。"""
    namespace_state = state.get(namespace) or {}
    if not isinstance(namespace_state, dict):
        return 0
    return int(namespace_state.get(key, 0) or 0)

def update_nested_count(
    state: dict[str, Any],
    namespace: str,
    key: str,
    value: int,
) -> dict[str, dict[str, int]]:
    """更新 State 嵌套命名空间字典中的计数值，并返回状态更新字典。"""
    namespace_state = state.get(namespace) or {}
    if not isinstance(namespace_state, dict):
        namespace_state = {}
    return {namespace: {**namespace_state, key: value}}

def make_named_system_message(name: str, text: str) -> SystemMessage:
    """构造一个带 name 属性的 SystemMessage（用于指定提示词插槽）。"""
    slot_name = name.strip()
    if not slot_name:
        raise ValueError("SystemMessage 的 name 属性不能为空")
    return SystemMessage(content=text, name=slot_name)

def remove_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
) -> list[AnyMessage]:
    """从消息列表中移除指定 name 的 SystemMessage。"""
    output: list[AnyMessage] = []
    for message in messages or []:
        if isinstance(message, SystemMessage) and message.name == name:
            continue
        output.append(message)
    return output

def upsert_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
    text: str,
) -> list[AnyMessage]:
    """向消息列表中更新或插入指定 name 的 SystemMessage。"""
    replacement = make_named_system_message(name, text)
    output: list[AnyMessage] = []
    replaced = False
    inserted = False

    for message in messages or []:
        if isinstance(message, SystemMessage):
            if message.name == replacement.name:
                if not replaced:
                    output.append(replacement)
                    replaced = True
                continue
            output.append(message)
            continue

        if not inserted:
            if not replaced:
                output.append(replacement)
                replaced = True
            inserted = True
        output.append(message)

    if not replaced:
        output.append(replacement)
    return output

def set_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
    text: str | None,
) -> list[AnyMessage]:
    """快捷设置命名 SystemMessage，如果文本为空则执行移除。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return remove_named_system_message(messages, name=name)
    return upsert_named_system_message(messages, name=name, text=cleaned)
