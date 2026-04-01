from __future__ import annotations

import sys
import warnings

import torch
from torch import Tensor


def psd_safe_cholesky(
    mat: Tensor,
    jitter: float = 1e-6,
    max_retries: int = 3,
) -> Tensor:
    """Cholesky decomposition with adaptive jitter on failure.

    Tries the raw matrix first.  On ``RuntimeError`` (non-PD),
    adds geometrically increasing diagonal jitter until the
    decomposition succeeds or *max_retries* is exhausted.
    """
    try:
        return torch.linalg.cholesky(mat)
    except RuntimeError:
        eye = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
        current = jitter
        for _ in range(max_retries):
            try:
                result = torch.linalg.cholesky(mat + current * eye)
                warnings.warn(
                    f"psd_safe_cholesky: added jitter={current:.1e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return result
            except RuntimeError:
                current *= 10.0
        raise RuntimeError(
            f"Matrix not positive-definite after {max_retries} "
            f"jitter retries (max jitter={current:.1e})"
        )


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
