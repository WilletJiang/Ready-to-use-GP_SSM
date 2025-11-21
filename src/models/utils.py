from __future__ import annotations

import sys

import torch


def torch_compile(func):
    """Decorator that opts into torch.compile or torch.jit when available."""

    if sys.version_info >= (3, 13):
        try:
            return torch.jit.script(func)
        except Exception:
            return func
    try:
        from torch import compile as _compile

        try:
            return _compile(func)
        except Exception:
            return func
    except ImportError:
        return func

