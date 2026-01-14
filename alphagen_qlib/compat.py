"""Compatibility helpers for qlib and newer numpy/joblib."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def patch_numpy_loadtxt_for_qlib_calendar() -> None:
    """Patch numpy.loadtxt so qlib calendar reading tolerates newline delimiters."""
    if hasattr(np, "_real_loadtxt"):
        return

    np._real_loadtxt = np.loadtxt  # type: ignore[attr-defined]

    def _patched_loadtxt(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("delimiter") in ("\n", "\r"):
            kwargs = dict(kwargs)
            kwargs["delimiter"] = None
        return np._real_loadtxt(*args, **kwargs)  # type: ignore[attr-defined]

    np.loadtxt = _patched_loadtxt  # type: ignore[assignment]


def patch_qlib_parallelext_for_joblib() -> None:
    """Patch qlib ParallelExt for joblib maxtasksperchild compatibility."""
    try:
        from joblib import Parallel as JoblibParallel
        from joblib._parallel_backends import MultiprocessingBackend
    except Exception:
        return

    try:
        from qlib.utils.paral import ParallelExt
    except Exception:
        return

    if getattr(ParallelExt, "_patched_for_maxtasksperchild", False):
        return

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        maxtasksperchild = kwargs.get("maxtasksperchild")
        JoblibParallel.__init__(self, *args, **kwargs)

        backend = getattr(self, "_backend", None)
        if (
            backend is not None
            and isinstance(backend, MultiprocessingBackend)
            and maxtasksperchild is not None
        ):
            if isinstance(getattr(self, "_backend_kwargs", None), dict):
                self._backend_kwargs["maxtasksperchild"] = maxtasksperchild
            elif isinstance(getattr(self, "_backend_args", None), dict):
                self._backend_args["maxtasksperchild"] = maxtasksperchild
            else:
                self._backend_kwargs = {"maxtasksperchild": maxtasksperchild}

    ParallelExt.__init__ = _patched_init  # type: ignore[assignment]
    ParallelExt._patched_for_maxtasksperchild = True  # type: ignore[attr-defined]


def guard_qlib_init(provider_uri: str) -> None:
    """Guard qlib.init against later calls without or with different provider_uri."""
    try:
        import qlib
    except Exception:
        return

    if not hasattr(qlib, "_real_init"):
        qlib._real_init = qlib.init  # type: ignore[attr-defined]

    if getattr(qlib, "_guarded_init", False):
        return

    qlib._guarded_provider_uri = provider_uri  # type: ignore[attr-defined]

    def _guarded_init(*args: Any, **kwargs: Any) -> Optional[Any]:
        if "provider_uri" in kwargs:
            call_provider = kwargs["provider_uri"]
        elif len(args) >= 1:
            call_provider = args[0]
        else:
            call_provider = None

        if call_provider is None:
            return None

        if call_provider != qlib._guarded_provider_uri:  # type: ignore[attr-defined]
            print(
                "qlib.init ignored: already initialized with provider_uri="
                f"{qlib._guarded_provider_uri!r}"  # type: ignore[attr-defined]
            )
            return None

        return qlib._real_init(*args, **kwargs)  # type: ignore[attr-defined]

    qlib.init = _guarded_init  # type: ignore[assignment]
    qlib._guarded_init = True  # type: ignore[attr-defined]


def patch_all(provider_uri: str) -> None:
    """Apply all compatibility patches."""
    patch_numpy_loadtxt_for_qlib_calendar()
    patch_qlib_parallelext_for_joblib()
    guard_qlib_init(provider_uri)
