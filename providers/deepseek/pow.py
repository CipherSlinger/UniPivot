"""DeepSeek 网页端 Proof-of-Work 求解器。

DeepSeek 的 chat.deepseek.com 对补全接口启用了 PoW 校验（x-ds-pow-response）。
算法 "DeepSeekHashV1" 以 WebAssembly 形式下发（与浏览器加载的是同一份
sha3_wasm_bg.wasm），本模块在 wasmtime 沙箱（无文件/网络权限）中直接运行该
wasm，避免用 Python 重写其 float64 哈希逻辑。

实现参考: https://github.com/sums001/Deepseek-API （MIT License）
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Optional

import wasmtime

WASM_PATH = Path(__file__).resolve().parent / "sha3_wasm_bg.wasm"


class DeepSeekPow:
    """加载 wasm 一次，为每个 challenge 生成 x-ds-pow-response 头值。

    注意: wasmtime 的 Store 不可重入，并发调用需由调用方加锁串行化。
    """

    def __init__(self, wasm_path: Path = WASM_PATH):
        self._store = wasmtime.Store()
        module = wasmtime.Module.from_file(self._store.engine, str(wasm_path))
        self._inst = wasmtime.Instance(self._store, module, [])
        exp = self._inst.exports(self._store)
        self._memory: wasmtime.Memory = exp["memory"]
        self._solve = exp["wasm_solve"]
        self._malloc = exp["__wbindgen_export_0"]  # malloc(size, align)
        self._add_to_stack = exp["__wbindgen_add_to_stack_pointer"]

    def _write_str(self, text: str) -> tuple[int, int]:
        """malloc 并写入 UTF-8 字符串，返回 (ptr, len)。"""
        data = text.encode("utf-8")
        ptr = self._malloc(self._store, len(data), 1)
        base = self._memory.data_ptr(self._store)
        for i, b in enumerate(data):
            base[ptr + i] = b
        return ptr, len(data)

    def solve(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """求解 PoW，返回整数答案；wasm 报告失败时返回 None。

        对应网页端 wasm-bindgen 调用:
            wasm_solve(retptr, challenge_ptr, challenge_len,
                       prefix_ptr, prefix_len, difficulty)
        在 shadow stack 预留 16 字节返回槽：+0 为 i32 状态，+8 为 f64 答案。
        """
        retptr = self._add_to_stack(self._store, -16)
        try:
            c_ptr, c_len = self._write_str(challenge)
            p_ptr, p_len = self._write_str(prefix)
            self._solve(self._store, retptr, c_ptr, c_len, p_ptr, p_len, float(difficulty))

            mem = self._memory.data_ptr(self._store)
            status = struct.unpack("<i", bytes(mem[retptr:retptr + 4]))[0]
            value = struct.unpack("<d", bytes(mem[retptr + 8:retptr + 16]))[0]
        finally:
            self._add_to_stack(self._store, 16)

        if status == 0:
            return None
        return int(value)

    def make_header(self, challenge: dict) -> str:
        """根据 challenge 构造 base64 的 x-ds-pow-response 头值。"""
        prefix = f"{challenge['salt']}_{challenge['expire_at']}_"
        answer = self.solve(challenge["challenge"], prefix, challenge["difficulty"])
        if answer is None:
            raise RuntimeError("PoW 求解失败（挑战可能已过期），请重试")
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(raw).decode("utf-8")
