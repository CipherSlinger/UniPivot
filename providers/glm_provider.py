"""兼容别名模块：向后兼容，实现已迁移至 providers.glm.provider。"""
import sys
from .glm import provider as _provider

sys.modules[__name__] = _provider
