"""兼容别名模块：向后兼容，实现位于 providers.doubao.provider。"""
import sys
from .doubao import provider as _provider

sys.modules[__name__] = _provider
