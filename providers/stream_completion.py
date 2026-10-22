"""上游流完整性判据 (S2)。

一条上游流在**没有完成标记**的情况下结束，与「正常说完」在生成器层面长得
一模一样：五家的 `chat()` 结尾都无条件产出一帧末位元数据，协议处理器随后
无条件补上 `finish_reason: "stop"` 与 `data: [DONE]`。于是客户端收到的是一个
**自称完整**的半截回答，而节点的失败计数保持 0（`record_success` 记在首包，
耗尽路径上不再复核），下一轮它仍然是主选。

对 agent 来说这是最致命的一种：半截的 ```tool_call 围栏解析不出任何东西，
这一轮既没有答案也没有调用，而且没有任何错误可看。

判据只能覆盖**真的有标记**的协议
-----------------------------------
实测五家协议：

  * qwen  读 `data.status == "complete"`；
  * kimi  读 `event == "all_done"`；
  * deepseek / doubao / glm **不存在**被读取的完成标记（doubao 的 `is_finish`
    只在出站请求体里，glm 的 `finish` 一次都没出现）。

所以只有两家能判定截断。对另外三家编一个启发式（「文本不以标点结尾」之类）
就是凭空制造证据 —— 本仓库反复在删的正是这种东西。

判据因此是**三态**，而不是布尔：

    True   协议有标记且看到了     → 正常结束
    False  协议有标记且没看到     → 截断
    None   协议没有标记（未知）   → 行为零变化（fail-open）

「缺键 = 未知 ≠ 截断」是三家无标记上游继续正常工作的唯一保证，与
`resolve_allowed_tools` 对无法识别的 `tool_choice` 的处置是同一条惯例：
形状上 fail-open，语义上 fail-closed。
"""

from __future__ import annotations

import os
from typing import Any, Optional

# 末位元数据帧里携带完整性的键名。只有协议**真有**完成标记的节点才会设置它。
COMPLETION_KEY = "upstream_complete"


def completion_check_enabled() -> bool:
    """完整性判据是否生效（`GATEWAY_DISABLE_STREAM_COMPLETION_CHECK=1` 关闭）。"""
    return os.getenv("GATEWAY_DISABLE_STREAM_COMPLETION_CHECK") != "1"


def observe(meta: Any, current: Optional[bool]) -> Optional[bool]:
    """从一帧元数据里读取完整性，返回更新后的三态。

    只有显式的布尔值才会改变判据：缺键、非 dict、非布尔值全部保持 `current`
    不变（即「这一帧没有带完整性信息」，而不是「不完整」）。取**最后一次**
    看到的值 —— qwen 带重试循环，真正算数的是末位元数据帧产出时的状态。
    """
    if not isinstance(meta, dict):
        return current
    value = meta.get(COMPLETION_KEY)
    if isinstance(value, bool):
        return value
    return current


def is_incomplete(state: Optional[bool]) -> bool:
    """是否**确定**被截断。只有 `False` 算；`None`（未知）不算。"""
    return state is False
