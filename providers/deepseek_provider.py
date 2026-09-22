"""兼容别名模块：向后兼容，实现已迁移至 providers.deepseek.provider。"""
import sys
from .deepseek import provider as _provider

sys.modules[__name__] = _provider
