"""兼容别名模块：PoW 求解器已迁移至 providers.deepseek.pow。"""
import sys
from providers.deepseek import pow as _pow

sys.modules[__name__] = _pow
