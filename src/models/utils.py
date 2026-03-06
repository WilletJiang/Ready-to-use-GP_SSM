from __future__ import annotations

import sys
import warnings

import torch


def torch_compile(func):
    """Decorator that opts into torch.compile or torch.jit when available."""

    if sys.version_info >= (3, 13):
        try:
            return torch.jit.script(func)
        except Exception as exc:  # pragma: no cover - backend/version dependent
            warnings.warn(
                f"torch.jit.script failed ({exc}); falling back to eager function",
                RuntimeWarning,
            )
            return func
    try:
        from torch import compile as _compile

        try:
            compiled = _compile(func)
        except Exception as exc:  # pragma: no cover - backend/version dependent
            warnings.warn(
                f"torch.compile failed ({exc}); falling back to eager function",
                RuntimeWarning,
            )
            return func
        def _is_compile_error(err: Exception) -> bool:
            try:
                from torch._dynamo.exc import BackendCompilerFailed

                if isinstance(err, BackendCompilerFailed):
                    return True
            except Exception:
                pass
            name = err.__class__.__name__
            if "BackendCompilerFailed" in name:
                return True
            message = str(err)
            return "BackendCompilerFailed" in message or "InvalidCxxCompiler" in message

        def wrapped(*args, **kwargs):
            nonlocal compiled
            if compiled is None:
                return func(*args, **kwargs)
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - backend/runtime dependent
                if _is_compile_error(exc):
                    warnings.warn(
                        f"torch.compile failed at runtime ({exc}); falling back to eager function",
                        RuntimeWarning,
                    )
                    compiled = None
                    return func(*args, **kwargs)
                raise

        return wrapped
    except ImportError:
        return func
