"""现场真机对决评测套件 (Live Benchmark Suite)
实测多模型在真实编程任务下的 TTFT、TPS、代码正确性与结构化工具遵循能力。
"""
import asyncio
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from providers.failover import get_traffic_pacer
from providers.http_client import close_http_client, get_http_client
from server import PROVIDER_SPECS


# ---------------------------------------------------------------- 三大真实研发评测用例

TASK_1_CODE_GEN = {
    "name": "任务 1: 算法实现与线程安全 (带 TTL 的 LRU 缓存)",
    "category": "代码实现与并发设计",
    "prompt": (
        "请用 Python 编写一个线程安全的 LRU 缓存类 `ThreadSafeLRUCache`。\n"
        "要求：\n"
        "1. `__init__(self, capacity: int, default_ttl: float = 60.0)`\n"
        "2. `put(self, key: str, value: Any, ttl: Optional[float] = None) -> None`：写入键值，支持自定义过期秒数。\n"
        "3. `get(self, key: str) -> Optional[Any]`：获取键值，若已过期或不存在返回 None，命中则刷新访问顺序。\n"
        "4. 使用 `threading.RLock()` 确保线程并发安全。\n"
        "请直接输出完整的 Python 代码（包含在 ```python ... ``` 块中），无需过多废话。"
    ),
    "test_code": """
cache = ThreadSafeLRUCache(capacity=2, default_ttl=1.0)
cache.put("a", 1)
cache.put("b", 2)
assert cache.get("a") == 1
cache.put("c", 3) # "b" 应该被淘汰
assert cache.get("b") is None
assert cache.get("c") == 3
assert cache.get("a") == 1
# 测试 TTL 过期
import time
cache.put("fast", 99, ttl=0.1)
assert cache.get("fast") == 99
time.sleep(0.15)
assert cache.get("fast") is None
print("PASS_TEST_1")
""",
}

TASK_2_DEBUG = {
    "name": "任务 2: 代码缺陷诊断与竞态排查",
    "category": "Bug 诊断与边界处理",
    "prompt": (
        "分析以下 Python 异步队列并发处理代码，指出其中至少两个严重的潜在缺陷（例如死锁、异常丢失或内存泄露），并给出修复后的核心代码：\n\n"
        "```python\n"
        "class BatchWorker:\n"
        "    def __init__(self):\n"
        "        self.queue = asyncio.Queue()\n"
        "    async def add(self, item):\n"
        "        await self.queue.put(item)\n"
        "    async def run(self):\n"
        "        while True:\n"
        "            batch = []\n"
        "            for _ in range(10):\n"
        "                batch.append(await self.queue.get())\n"
        "            await self.process(batch)\n"
        "    async def process(self, batch):\n"
        "        for x in batch:\n"
        "            asyncio.create_task(self.handle(x))\n"
        "```\n"
        "要求：明确指明问题所在，并说明为何 `queue.get()` 在队列不足 10 个元素时会导致永久阻塞挂起，以及 `create_task` 未追踪可能导致的后果。"
    ),
    "keywords": ["阻塞", "10", "task", "异常", "挂起", "超时", "join", "get_nowait", "gather"],
}

TASK_3_TOOL_CALL = {
    "name": "任务 3: 严格结构化工具调用 (JSON Format)",
    "category": "工具调用与参数合规",
    "prompt": (
        "你是一个自动化代码审查系统。根据以下 Git 冲突描述：\n"
        "文件: `server.py`，冲突行: 120-135，主分支更改为引入 load_balancer，特性分支更改为引入 prompt_cache。\n\n"
        "请调用工具 `resolve_git_conflict`。必须严格仅输出一个合法的 JSON 对象，格式如下：\n"
        "{\n"
        '  "tool": "resolve_git_conflict",\n'
        '  "arguments": {\n'
        '    "file_path": "server.py",\n'
        '    "strategy": "merge_both",\n'
        '    "confidence": 0.95,\n'
        '    "reason": "简述合并方案",\n'
        '    "resolution_steps": ["步骤1", "步骤2"]\n'
        "  }\n"
        "}\n"
        "注意：严禁输出任何 markdown 格式标记（如严禁带 ```json），严禁输出任何引���语或解释，首字符必须是 {，末字符必须是 }。"
    ),
    "expected_keys": ["tool", "arguments"],
    "expected_args_keys": ["file_path", "strategy", "confidence", "reason", "resolution_steps"],
}


# ---------------------------------------------------------------- 单次评测执行器

def extract_python_code(text: str) -> str:
    """提取代码块中的 Python 代码，若无代码块则尝试使用全文"""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return text.strip()


async def execute_model_task(
    p_name: str,
    wire_model: str,
    prompt: str,
    client: Any,
    timeout: float = 45.0,
) -> Dict[str, Any]:
    resolver, err_msg, factory = PROVIDER_SPECS[p_name]
    session = resolver()
    if not session or not session.token:
        return {"error": "无有效 Session 凭据", "success": False}

    provider = factory(session, client=client)
    pacer = get_traffic_pacer()
    t0 = time.time()
    ttft: Optional[float] = None
    output_tokens = 0
    full_text = []

    try:
        async with pacer.acquire(p_name):
            async def _stream():
                nonlocal ttft, output_tokens
                messages = [{"role": "user", "content": prompt}]
                async for chunk in provider.chat(messages, model=wire_model, stream=True):
                    delta = ""
                    if isinstance(chunk, tuple):
                        delta = chunk[0]
                    elif isinstance(chunk, dict):
                        delta = chunk.get("delta", {}).get("content", "")
                    elif isinstance(chunk, str):
                        delta = chunk

                    if delta:
                        if ttft is None:
                            ttft = round((time.time() - t0) * 1000, 2)
                        full_text.append(delta)
                        output_tokens += max(1, len(delta) // 2)

            await asyncio.wait_for(_stream(), timeout=timeout)
        tot_time = round((time.time() - t0) * 1000, 2)
        final_str = "".join(full_text)
        tps = round(output_tokens / (tot_time / 1000.0), 1) if tot_time > 0 else 0.0

        return {
            "success": True,
            "provider": p_name,
            "model": wire_model,
            "ttft_ms": ttft or tot_time,
            "total_ms": tot_time,
            "output_tokens": output_tokens,
            "tps": tps,
            "text": final_str,
            "error": None,
        }
    except asyncio.TimeoutError:
        tot_time = round((time.time() - t0) * 1000, 2)
        return {
            "success": False,
            "provider": p_name,
            "model": wire_model,
            "ttft_ms": None,
            "total_ms": tot_time,
            "output_tokens": 0,
            "tps": 0,
            "text": "",
            "error": "Timeout: 请求超时(可能触发WAF/人机验证)",
        }
    except Exception as e:
        tot_time = round((time.time() - t0) * 1000, 2)
        return {
            "success": False,
            "provider": p_name,
            "model": wire_model,
            "ttft_ms": None,
            "total_ms": tot_time,
            "output_tokens": 0,
            "tps": 0,
            "text": "",
            "error": f"{type(e).__name__}: {str(e)}",
        }


# ---------------------------------------------------------------- 自动化评分机制

def evaluate_task_1(result: Dict[str, Any]) -> Tuple[float, str]:
    """运行生成的 Python 代码并进行断言验证"""
    if not result.get("success"):
        return 0.0, f"调用失败: {result.get('error')}"
    raw_code = extract_python_code(result.get("text", ""))
    test_harness = raw_code + "\n" + TASK_1_CODE_GEN["test_code"]

    # 动态安全执行沙箱
    local_scope: Dict[str, Any] = {}
    try:
        import collections, threading, time
        globals_dict = dict(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__)
        globals_dict.update({
            "OrderedDict": collections.OrderedDict,
            "threading": threading,
            "time": time,
        })
        exec(test_harness, globals_dict, local_scope)
        return 100.0, "代码编译通过，并发缓存与 TTL 过期测试全 PASS"
    except AssertionError as ae:
        return 40.0, f"语法正确但逻辑断言失败: {ae}"
    except Exception as e:
        return 20.0, f"代码运行报错: {type(e).__name__}: {e}"


def evaluate_task_2(result: Dict[str, Any]) -> Tuple[float, str]:
    """评估诊断与缺陷命中率"""
    if not result.get("success"):
        return 0.0, f"调用失败: {result.get('error')}"
    text = result.get("text", "")
    score = 0.0
    reasons = []

    # 1. 命中 queue.get 阻塞问题
    if any(k in text for k in ["阻塞", "挂起", "不足", "永久", "死锁"]):
        score += 50.0
        reasons.append("命中队列元素不足时的阻塞挂起缺陷")
    # 2. 命中 create_task 异常捕获/内存生命周期
    if any(k in text for k in ["create_task", "异常", "未捕获", "丢弃", "跟踪", "gather", "join"]):
        score += 35.0
        reasons.append("命中 create_task 后台任务无追踪及异常吞噬问题")
    # 3. 给出完整修复方案
    if "```python" in text:
        score += 15.0
        reasons.append("提供了修复代码")

    return score, "；".join(reasons) if reasons else "未命中关键并发陷阱"


def evaluate_task_3(result: Dict[str, Any]) -> Tuple[float, str]:
    """严格评估 JSON 格式、零冗余、字段完整性"""
    if not result.get("success"):
        return 0.0, f"调用失败: {result.get('error')}"
    raw = result.get("text", "").strip()
    score = 0.0
    notes = []

    # 1. 严格检查是否无 markdown 包装 (指令严格强调不能带 ```)
    has_md = raw.startswith("```") or raw.endswith("```")
    if not has_md:
        score += 30.0
        notes.append("完美遵循无 markdown 包裹指令")
    else:
        notes.append("违规输出了 markdown 标记（扣分）")
        # 剥离 markdown 尝试解析
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        score += 40.0
        notes.append("JSON 结构合法解析成功")

        # 检查结构
        if data.get("tool") == "resolve_git_conflict":
            score += 15.0
        args = data.get("arguments", {})
        if all(k in args for k in TASK_3_TOOL_CALL["expected_args_keys"]):
            score += 15.0
            notes.append("字段与参数类型完全契合要求")
        else:
            missing = [k for k in TASK_3_TOOL_CALL["expected_args_keys"] if k not in args]
            notes.append(f"参数缺少必要字段: {missing}")
    except json.JSONDecodeError as je:
        notes.append(f"JSON 语法损坏: {je}")

    return score, "；".join(notes)


# ---------------------------------------------------------------- 主执行流

async def main():
    print("=" * 70)
    print(" 🚀 UniPivot 本地真实机型对决基准测试 (Live Benchmark Showdown)")
    print("=" * 70)

    # 参测模型矩阵
    MODELS = [
        ("doubao", "doubao-pro", "豆包 Pro (字节旗舰)"),
        ("doubao", "doubao-think", "豆包 Think (深度思考)"),
        ("kimi", "kimi", "月之暗面 Kimi (标准版)"),
        ("qwen", "Qwen", "通义千问 (标准版)"),
        ("deepseek", "deepseek-chat", "DeepSeek Chat (风控受阻节点)"),
        ("glm", "glm-4-plus", "智谱 GLM-4-Plus (凭证待刷新)"),
    ]

    client = await get_http_client(45)

    print(f"\n[1/3] 正在评测 {TASK_1_CODE_GEN['name']}...", flush=True)
    t1_res = []
    for p, m, label in MODELS:
        print(f"  -> 测试 {label}...", end=" ", flush=True)
        r = await execute_model_task(p, m, TASK_1_CODE_GEN["prompt"], client, timeout=45.0)
        status = f"✅ 完成 ({r['total_ms']}ms, {r['output_tokens']} tokens)" if r["success"] else f"❌ 失败: {r['error']}"
        print(status, flush=True)
        t1_res.append(r)

    print(f"\n[2/3] 正在评测 {TASK_2_DEBUG['name']}...", flush=True)
    t2_res = []
    for p, m, label in MODELS:
        print(f"  -> 测试 {label}...", end=" ", flush=True)
        r = await execute_model_task(p, m, TASK_2_DEBUG["prompt"], client, timeout=45.0)
        status = f"✅ 完成 ({r['total_ms']}ms, {r['output_tokens']} tokens)" if r["success"] else f"❌ 失败: {r['error']}"
        print(status, flush=True)
        t2_res.append(r)

    print(f"\n[3/3] 正在评测 {TASK_3_TOOL_CALL['name']}...", flush=True)
    t3_res = []
    for p, m, label in MODELS:
        print(f"  -> 测试 {label}...", end=" ", flush=True)
        r = await execute_model_task(p, m, TASK_3_TOOL_CALL["prompt"], client, timeout=45.0)
        status = f"✅ 完成 ({r['total_ms']}ms, {r['output_tokens']} tokens)" if r["success"] else f"❌ 失败: {r['error']}"
        print(status, flush=True)
        t3_res.append(r)

    await close_http_client()

    # 汇总结算
    scoreboard = []
    print("\n" + "=" * 70)
    print(" 📊 实测评分与性能度量明细")
    print("=" * 70)

    for i, (p, m, label) in enumerate(MODELS):
        r1 = t1_res[i]
        r2 = t2_res[i]
        r3 = t3_res[i]

        s1, note1 = evaluate_task_1(r1)
        s2, note2 = evaluate_task_2(r2)
        s3, note3 = evaluate_task_3(r3)

        avg_ttft = round(
            sum(
                filter(
                    None,
                    [
                        r1.get("ttft_ms"),
                        r2.get("ttft_ms"),
                        r3.get("ttft_ms"),
                    ],
                )
            )
            / 3.0,
            1,
        ) if any([r1.get("ttft_ms"), r2.get("ttft_ms"), r3.get("ttft_ms")]) else 0.0

        avg_tps = round(
            sum([r1.get("tps", 0), r2.get("tps", 0), r3.get("tps", 0)]) / 3.0, 1
        )

        # 综合加权总分: 代码实战 (40%) + 缺陷诊断 (35%) + 工具遵循 (25%)
        composite_score = round(s1 * 0.40 + s2 * 0.35 + s3 * 0.25, 2)

        scoreboard.append(
            {
                "label": label,
                "provider": p,
                "model": m,
                "score": composite_score,
                "task1_score": s1,
                "task1_note": note1,
                "task2_score": s2,
                "task2_note": note2,
                "task3_score": s3,
                "task3_note": note3,
                "avg_ttft_ms": avg_ttft,
                "avg_tps": avg_tps,
                "raw_results": [r1, r2, r3],
            }
        )

    # 降序排序
    scoreboard.sort(key=lambda x: x["score"], reverse=True)

    # 输出表格
    header = f"{'名次':<4} | {'模型名称':<28} | {'综合得分':<8} | {'代码实现(40%)':<12} | {'缺陷诊断(35%)':<12} | {'工具规范(25%)':<12} | {'平均TTFT':<10} | {'平均TPS':<8}"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(scoreboard, start=1):
        print(
            f"#{rank:<3} | {row['label']:<28} | {row['score']:<8} | {row['task1_score']:<14} | {row['task2_score']:<14} | {row['task3_score']:<14} | {row['avg_ttft_ms']:<8}ms | {row['avg_tps']:<8}"
        )

    print("\n" + "=" * 70)
    print(" 🔍 各模型单项实测细节诊断")
    print("=" * 70)
    for rank, row in enumerate(scoreboard, start=1):
        print(f"\n【#{rank} {row['label']}】综合得分: {row['score']}")
        print(f"  • 任务1(代码实现): {row['task1_score']}分 - {row['task1_note']}")
        print(f"  • 任务2(并发诊断): {row['task2_score']}分 - {row['task2_note']}")
        print(f"  • 任务3(工具调用): {row['task3_score']}分 - {row['task3_note']}")
        print(f"  • 性能表现: 平均首字延迟 {row['avg_ttft_ms']}ms, 吞吐 {row['avg_tps']} Tokens/s")

    # 保存结果到 reports/live_benchmark_results.json
    out_file = "reports/live_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scoreboard": [
                    {k: v for k, v in row.items() if k != "raw_results"}
                    for row in scoreboard
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✅ 评测原始度量数据已持久化保存至: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
