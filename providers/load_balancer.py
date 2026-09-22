"""冷热模型动态负载均衡与自适应降级分流矩阵 (providers/load_balancer.py)。

实现 [P0] OPT-003：
1. 滑动窗口统计器 SlidingWindowMetrics：
   - 按时间窗口（默认 60s）以秒级桶 (Buckets) 追踪各 Provider 实时指标：
     - request_count (当前窗口请求总数)
     - success_count 与 error_count
     - active_concurrency (当前并发活动连接数)
     - avg_ttft_ms (平均首包延迟)
     - rate_limit_hits (429/WAF 阻断命中数)
2. 自适应负载均衡调度器 AdaptiveLoadBalancer：
   - 高水位避让与动态分流 (High Watermark Avoidance & Shedding)：
     高风控 Provider 并发接近上限或近期错误率上升时，在同能力环中打散分流到宽松低负载节点。
   - 轻量请求自适应降级 (Query Complexity Degradation)：
     短 Prompt (如 < 300 tokens) 且无思考需求时，若旗舰环高负载，自适应降级至 SPEED 环高速经济模型。
   - 状态导出方法 get_load_stats()：提供诊断接口与前端可视化数据。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .failover import (
    CapabilityRing,
    CooldownTracker,
    PROVIDER_RISK_SPECS,
    ProviderRiskMeta,
    RING_DEFINITIONS,
    RiskLevel,
    classify_capability_ring,
)

logger = logging.getLogger("gateway.load_balancer")


@dataclass
class Bucket:
    """秒级时间桶数据"""
    requests: int = 0
    successes: int = 0
    errors: int = 0
    rate_limits: int = 0
    ttft_sum_ms: float = 0.0
    ttft_count: int = 0


@dataclass
class ProviderMetricsSnapshot:
    """提供方滑动窗口度量快照"""
    provider_name: str
    request_count: int
    success_count: int
    error_count: int
    rate_limit_hits: int
    active_concurrency: int
    avg_ttft_ms: float
    qps: float
    error_rate: float
    shed_count: int
    shed_received_count: int
    degraded_count: int
    degraded_received_count: int

    def to_dict(self) -> dict:
        return {
            "provider": self.provider_name,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "rate_limit_hits": self.rate_limit_hits,
            "active_concurrency": self.active_concurrency,
            "avg_ttft_ms": self.avg_ttft_ms,
            "avg_ttft": self.avg_ttft_ms,  # 别名兼容
            "qps": self.qps,
            "error_rate": self.error_rate,
            "shed_count": self.shed_count,
            "shed_received_count": self.shed_received_count,
            "degraded_count": self.degraded_count,
            "degraded_received_count": self.degraded_received_count,
        }


class SlidingWindowMetrics:
    """基于秒级桶的滑动窗口统计器（线程安全，支持模拟时间推移测试）"""

    def __init__(self, provider_name: str, window_seconds: float = 60.0):
        self.provider_name = provider_name
        self.window_seconds = float(window_seconds)
        # 秒级时间戳 -> Bucket
        self._buckets: Dict[int, Bucket] = {}
        self._active_concurrency: int = 0
        self._shed_count: int = 0
        self._shed_received_count: int = 0
        self._degraded_count: int = 0
        self._degraded_received_count: int = 0
        self._lock = threading.Lock()

    @property
    def active_concurrency(self) -> int:
        with self._lock:
            return self._active_concurrency

    def _cleanup(self, current_time: float) -> None:
        """清理已超出窗口的陈旧桶"""
        cutoff = int(current_time - self.window_seconds)
        expired_keys = [ts for ts in self._buckets.keys() if ts <= cutoff]
        for k in expired_keys:
            del self._buckets[k]

    def _get_bucket(self, second_ts: int) -> Bucket:
        b = self._buckets.get(second_ts)
        if b is None:
            b = Bucket()
            self._buckets[second_ts] = b
        return b

    def acquire(self, now: Optional[float] = None) -> int:
        """记录请求开始并原子增加活跃并发计数。返回增加后的 active_concurrency。"""
        with self._lock:
            t = now if now is not None else time.time()
            self._active_concurrency += 1
            sec = int(t)
            bucket = self._get_bucket(sec)
            bucket.requests += 1
            self._cleanup(t)
            return self._active_concurrency

    def release(
        self,
        success: bool = True,
        is_rate_limit: bool = False,
        now: Optional[float] = None,
    ) -> int:
        """记录请求结束并原子减少活跃并发计数（下限保底 0）。返回减少后的 active_concurrency。"""
        with self._lock:
            t = now if now is not None else time.time()
            self._active_concurrency = max(0, self._active_concurrency - 1)
            sec = int(t)
            bucket = self._get_bucket(sec)
            if success:
                bucket.successes += 1
            else:
                bucket.errors += 1
                if is_rate_limit:
                    bucket.rate_limits += 1
            self._cleanup(t)
            return self._active_concurrency

    def record_ttft(self, ttft_ms: float, now: Optional[float] = None) -> None:
        """记录首包延迟 (TTFT)"""
        with self._lock:
            t = now if now is not None else time.time()
            sec = int(t)
            bucket = self._get_bucket(sec)
            bucket.ttft_sum_ms += max(0.0, float(ttft_ms))
            bucket.ttft_count += 1
            self._cleanup(t)

    def record_shed(self, as_source: bool = True) -> None:
        """记录主动分流事件（as_source: 作为分流源还是目标接收方）"""
        with self._lock:
            if as_source:
                self._shed_count += 1
            else:
                self._shed_received_count += 1

    def record_degraded(self, as_source: bool = True) -> None:
        """记录降级事件"""
        with self._lock:
            if as_source:
                self._degraded_count += 1
            else:
                self._degraded_received_count += 1

    def get_snapshot(self, now: Optional[float] = None) -> ProviderMetricsSnapshot:
        """获取当前滑动窗口内的聚合度量快照"""
        with self._lock:
            t = now if now is not None else time.time()
            self._cleanup(t)
            cutoff = int(t - self.window_seconds)

            req_cnt = 0
            succ_cnt = 0
            err_cnt = 0
            rl_cnt = 0
            ttft_sum = 0.0
            ttft_cnt = 0

            for sec, b in self._buckets.items():
                if sec > cutoff:
                    req_cnt += b.requests
                    succ_cnt += b.successes
                    err_cnt += b.errors
                    rl_cnt += b.rate_limits
                    ttft_sum += b.ttft_sum_ms
                    ttft_cnt += b.ttft_count

            avg_ttft = round(ttft_sum / ttft_cnt, 1) if ttft_cnt > 0 else 0.0
            qps = round(req_cnt / self.window_seconds, 2)
            error_rate = round(err_cnt / req_cnt, 3) if req_cnt > 0 else 0.0

            return ProviderMetricsSnapshot(
                provider_name=self.provider_name,
                request_count=req_cnt,
                success_count=succ_cnt,
                error_count=err_cnt,
                rate_limit_hits=rl_cnt,
                active_concurrency=self._active_concurrency,
                avg_ttft_ms=avg_ttft,
                qps=qps,
                error_rate=error_rate,
                shed_count=self._shed_count,
                shed_received_count=self._shed_received_count,
                degraded_count=self._degraded_count,
                degraded_received_count=self._degraded_received_count,
            )

    def reset(self) -> None:
        """重置所有统计度量与计数器"""
        with self._lock:
            self._buckets.clear()
            self._active_concurrency = 0
            self._shed_count = 0
            self._shed_received_count = 0
            self._degraded_count = 0
            self._degraded_received_count = 0


@dataclass
class BalancingDecision:
    """负载均衡与降级分流决策结果"""
    provider_key: str
    wire_model: str
    action: str  # "normal" | "shedded" | "degraded"
    original_provider: str
    original_model: str
    ring: CapabilityRing
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "provider_key": self.provider_key,
            "wire_model": self.wire_model,
            "action": self.action,
            "original_provider": self.original_provider,
            "original_model": self.original_model,
            "ring": self.ring.value,
            "reason": self.reason,
        }


class AdaptiveLoadBalancer:
    """五大模型冷热动态负载均衡与自适应降级调度引擎"""

    def __init__(
        self,
        cooldown_tracker: Optional[CooldownTracker] = None,
        risk_specs: Optional[Dict[str, ProviderRiskMeta]] = None,
        ring_definitions: Optional[Dict[CapabilityRing, List[Tuple[str, str]]]] = None,
        window_seconds: float = 60.0,
        high_watermark_ratio: float = 0.8,
        high_error_rate_threshold: float = 0.25,
        prompt_complexity_threshold: int = 300,
    ):
        self.cooldown_tracker = cooldown_tracker or CooldownTracker()
        self.risk_specs = risk_specs or PROVIDER_RISK_SPECS
        self.ring_definitions = ring_definitions or RING_DEFINITIONS
        self.window_seconds = float(window_seconds)
        self.high_watermark_ratio = float(high_watermark_ratio)
        self.high_error_rate_threshold = float(high_error_rate_threshold)
        self.prompt_complexity_threshold = int(prompt_complexity_threshold)

        self._metrics: Dict[str, SlidingWindowMetrics] = {
            pk: SlidingWindowMetrics(pk, window_seconds=self.window_seconds)
            for pk in self.risk_specs.keys()
        }
        self._lock = threading.Lock()

    def get_metrics(self, provider_key: str) -> SlidingWindowMetrics:
        """获取或懒加载指定节点的滑动窗口统计器"""
        with self._lock:
            if provider_key not in self._metrics:
                self._metrics[provider_key] = SlidingWindowMetrics(
                    provider_key, window_seconds=self.window_seconds
                )
            return self._metrics[provider_key]

    def acquire(self, provider_key: str, now: Optional[float] = None) -> int:
        """记录进入调用并递增活跃并发"""
        return self.get_metrics(provider_key).acquire(now=now)

    def release(
        self,
        provider_key: str,
        success: bool = True,
        is_rate_limit: bool = False,
        error_msg: str = "",
        now: Optional[float] = None,
    ) -> int:
        """记录完成调用并递减活跃并发"""
        return self.get_metrics(provider_key).release(
            success=success, is_rate_limit=is_rate_limit, now=now
        )

    def record_ttft(self, provider_key: str, ttft_ms: float, now: Optional[float] = None) -> None:
        """记录首包延迟"""
        self.get_metrics(provider_key).record_ttft(ttft_ms=ttft_ms, now=now)

    def is_high_load(
        self, provider_key: str, now: Optional[float] = None
    ) -> Tuple[bool, str]:
        """判定指定提供方是否处于高风控水位或高负载状态。
        返回 (is_high_load, reason)
        """
        # 1. 冷却期检查
        if self.cooldown_tracker.is_in_cooldown(provider_key):
            rem = self.cooldown_tracker.get_remaining_cooldown(provider_key)
            return True, f"节点处于冷却避让期 (剩余 {rem:.1f}s)"

        meta = self.risk_specs.get(provider_key)
        max_conc = meta.max_concurrency if meta else 4
        snap = self.get_metrics(provider_key).get_snapshot(now=now)

        # 2. 高水位并发检查 (active_concurrency >= max_concurrency * high_watermark_ratio)
        conc_threshold = max_conc * self.high_watermark_ratio
        if snap.active_concurrency >= conc_threshold:
            return True, (
                f"并发度过高 ({snap.active_concurrency}/{max_conc} >= "
                f"{int(self.high_watermark_ratio * 100)}% 水位)"
            )

        # 3. 错误率与风控阻断检查
        if snap.rate_limit_hits > 0:
            return True, f"近期命中 {snap.rate_limit_hits} 次 429/WAF 频控阻断"

        if snap.request_count >= 3 and snap.error_rate >= self.high_error_rate_threshold:
            return True, f"滑动窗口错误率上升至 {snap.error_rate * 100:.1f}%"

        return False, ""

    def select_route(
        self,
        primary_provider: str,
        primary_model: str,
        prompt_tokens: int = 0,
        is_thinking: bool = False,
        now: Optional[float] = None,
    ) -> BalancingDecision:
        """自适应负载均衡路由决策。

        流程：
        1. 轻量请求自适应降级：
           若请求目标为 FLAGSHIP 环，prompt_tokens < 300 且不需要思考推理，且 FLAGSHIP 目标高负载，
           自适应降级到 SPEED 环中的高速经济模型（如 Qwen3.6-Flash, doubao-lite, glm-4-flash）。
        2. 高水位避让与动态分流：
           若目标节点并发接近上限或近期错误率上升，在同能力环中打散分流到健康度高、并发度低的宽松节点。
        3. 正常路由：
           若主节点健康且负载在安全水位，走原始模型路由。
        """
        ring = classify_capability_ring(primary_provider, primary_model)

        # ------------------------------------------------ 1. 轻量请求自适应降级 (Query Complexity Degradation)
        is_lightweight = (
            ring == CapabilityRing.FLAGSHIP
            and prompt_tokens < self.prompt_complexity_threshold
            and not is_thinking
        )

        primary_busy, busy_reason = self.is_high_load(primary_provider, now=now)

        if is_lightweight and primary_busy:
            speed_candidates = self.ring_definitions.get(CapabilityRing.SPEED, [])
            best_speed = self._pick_best_candidate(
                candidates=speed_candidates,
                preferred_provider=primary_provider,
                now=now,
            )
            if best_speed:
                speed_pk, speed_model = best_speed
                self.get_metrics(primary_provider).record_degraded(as_source=True)
                self.get_metrics(speed_pk).record_degraded(as_source=False)
                reason = (
                    f"轻量请求 (tokens={prompt_tokens} < {self.prompt_complexity_threshold}, 无思考)，"
                    f"旗舰节点 [{primary_provider}] 高负载 ({busy_reason})，自适应降级至 SPEED 环 [{speed_pk}:{speed_model}]"
                )
                logger.info(f"[负载均衡] 自适应降级: {primary_provider}:{primary_model} -> {speed_pk}:{speed_model}")
                return BalancingDecision(
                    provider_key=speed_pk,
                    wire_model=speed_model,
                    action="degraded",
                    original_provider=primary_provider,
                    original_model=primary_model,
                    ring=CapabilityRing.SPEED,
                    reason=reason,
                )

        # ------------------------------------------------ 2. 高水位避让与动态分流 (Same-Ring Shedding)
        if primary_busy:
            same_ring_candidates = [
                (pk, wm)
                for pk, wm in self.ring_definitions.get(ring, [])
                if pk != primary_provider
            ]
            best_alt = self._pick_best_candidate(
                candidates=same_ring_candidates,
                preferred_provider=None,
                now=now,
            )
            if best_alt:
                alt_pk, alt_model = best_alt
                self.get_metrics(primary_provider).record_shed(as_source=True)
                self.get_metrics(alt_pk).record_shed(as_source=False)
                reason = (
                    f"节点 [{primary_provider}] 高负载/风控避让 ({busy_reason})，"
                    f"同环打散分流至健康宽松节点 [{alt_pk}:{alt_model}]"
                )
                logger.info(f"[负载均衡] 高水位分流: {primary_provider}:{primary_model} -> {alt_pk}:{alt_model}")
                return BalancingDecision(
                    provider_key=alt_pk,
                    wire_model=alt_model,
                    action="shedded",
                    original_provider=primary_provider,
                    original_model=primary_model,
                    ring=ring,
                    reason=reason,
                )

        # ------------------------------------------------ 3. 正常路由
        return BalancingDecision(
            provider_key=primary_provider,
            wire_model=primary_model,
            action="normal",
            original_provider=primary_provider,
            original_model=primary_model,
            ring=ring,
            reason="目标节点负载正常",
        )

    def _pick_best_candidate(
        self,
        candidates: List[Tuple[str, str]],
        preferred_provider: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Optional[Tuple[str, str]]:
        """从候选列表中挑选健康度最高、并发最低的最优节点"""
        available = []
        for pk, wm in candidates:
            # 排除处于冷却避让期的节点
            if self.cooldown_tracker.is_in_cooldown(pk):
                continue

            meta = self.risk_specs.get(pk)
            max_conc = meta.max_concurrency if meta else 4
            snap = self.get_metrics(pk).get_snapshot(now=now)

            # 排除同样处于高水位的节点
            if snap.active_concurrency >= max_conc * self.high_watermark_ratio:
                continue

            # 排除近期频控报错严重的节点
            if snap.rate_limit_hits > 0 or (snap.request_count >= 3 and snap.error_rate >= 0.5):
                continue

            load_ratio = snap.active_concurrency / max_conc if max_conc > 0 else 0.0

            # 风险评级权重：LOW=0, MEDIUM=1, HIGH=2
            risk_val = (
                0
                if meta and meta.risk_level == RiskLevel.LOW
                else (1 if meta and meta.risk_level == RiskLevel.MEDIUM else 2)
            )

            # 优先度排序键：
            # (risk_val, load_ratio, active_concurrency, error_rate)
            # 宽松节点（Doubao, GLM）risk_val=0，优先排前
            available.append((
                risk_val,
                load_ratio,
                snap.active_concurrency,
                snap.error_rate,
                pk,
                wm,
            ))

        if not available:
            return None

        available.sort()
        best = available[0]
        return (best[4], best[5])

    def get_load_stats(self, now: Optional[float] = None) -> dict:
        """导出全量负载均衡度量统计，供 /v1/diagnostics 与前端监控大屏展示"""
        t = now if now is not None else time.time()
        nodes_stat = {}
        total_active = 0
        total_requests = 0
        total_shed = 0
        total_degraded = 0

        for pk, meta in self.risk_specs.items():
            metrics = self.get_metrics(pk)
            snap = metrics.get_snapshot(now=t)
            max_conc = meta.max_concurrency
            conc_ratio = round(snap.active_concurrency / max_conc, 2) if max_conc > 0 else 0.0

            nodes_stat[pk] = {
                "provider": pk,
                "risk_level": meta.risk_level.value.upper(),
                "active_concurrency": snap.active_concurrency,
                "max_concurrency": max_conc,
                "concurrency_ratio": conc_ratio,
                "is_high_watermark": bool(snap.active_concurrency >= max_conc * self.high_watermark_ratio),
                "in_cooldown": self.cooldown_tracker.is_in_cooldown(pk),
                "remaining_cooldown": round(self.cooldown_tracker.get_remaining_cooldown(pk), 1),
                "request_count": snap.request_count,
                "success_count": snap.success_count,
                "error_count": snap.error_count,
                "rate_limit_hits": snap.rate_limit_hits,
                "error_rate": snap.error_rate,
                "qps": snap.qps,
                "avg_ttft_ms": snap.avg_ttft_ms,
                "avg_ttft": snap.avg_ttft_ms,
                "shed_count": snap.shed_count,
                "shed_received_count": snap.shed_received_count,
                "degraded_count": snap.degraded_count,
                "degraded_received_count": snap.degraded_received_count,
            }
            total_active += snap.active_concurrency
            total_requests += snap.request_count
            total_shed += snap.shed_count
            total_degraded += snap.degraded_count

        total_qps = round(total_requests / self.window_seconds, 2)

        return {
            "status": "active",
            "window_seconds": self.window_seconds,
            "high_watermark_ratio": self.high_watermark_ratio,
            "prompt_complexity_threshold": self.prompt_complexity_threshold,
            "total_active_concurrency": total_active,
            "total_qps": total_qps,
            "total_shed_count": total_shed,
            "total_degraded_count": total_degraded,
            "nodes": nodes_stat,
        }

    def reset(self) -> None:
        """重置所有节点的统计指标与并发计数"""
        with self._lock:
            for m in self._metrics.values():
                m.reset()
