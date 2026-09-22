"""兼容别名模块：向后兼容，实现已迁移至 providers.kimi.provider。"""
import sys
from .kimi import provider as _provider

sys.modules[__name__] = _provider
