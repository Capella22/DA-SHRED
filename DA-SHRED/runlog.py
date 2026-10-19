"""Run bookkeeping: timestamped output directories and session transcripts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
RESULTS = CODE / "results"


def quiet_blas_warnings(verbose=True):
    """Suppress spurious Accelerate/vecLib matmul FP-flag warnings."""
    import warnings
    for msg in ("divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul"):
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
    if verbose:
        print("  [runlog] suppressing spurious BLAS matmul FP-flag warnings; "
              "finiteness is asserted explicitly instead")


def check_finite(x, what="array"):
    """Raise if `x` contains NaN or Inf. Use wherever a spurious FP warning could otherwise mask a real numerical failure."""
    import numpy as np
    a = np.asarray(x)
    if not np.all(np.isfinite(a)):
        n_nan = int(np.isnan(a).sum()); n_inf = int(np.isinf(a).sum())
        raise FloatingPointError(
            f"{what} is not finite: {n_nan} NaN, {n_inf} Inf "
            f"(shape {a.shape}). This is a real numerical failure, not the "
            f"spurious Accelerate matmul warning.")
    return x


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, d):
        self.stream.write(d); self.stream.flush()
        self.fh.write(d); self.fh.flush()

    def flush(self):
        self.stream.flush(); self.fh.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


class RunDir:
    """Timestamped results directory with a live transcript."""

    def __init__(self, name, config=None, root=None, echo=True):
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.path = Path(root or RESULTS) / f"{name}_{self.stamp}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path / "session.txt", "w")
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, self._fh)
        sys.stderr = _Tee(sys.stderr, self._fh)
        self.write_json("env.json", _env())
        quiet_blas_warnings(verbose=False)
        if config is not None:
            self.write_json("config.json", config)
        if echo:
            print("=" * 70)
            print(f"{name}   {datetime.now().isoformat(timespec='seconds')}")
            print(f"results: {self.path}")
            print("=" * 70, flush=True)

    def file(self, *parts):
        p = self.path.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, fname, obj):
        with open(self.file(fname), "w") as f:
            json.dump(obj, f, indent=2, default=_default)

    def save_npz(self, fname, **arrays):
        import numpy as np
        np.savez_compressed(self.file(fname), **arrays)

    def close(self, status="ok"):
        print("=" * 70)
        print(f"status: {status}")
        print(f"results: {self.path}")
        print("=" * 70)
        sys.stdout, sys.stderr = self._out, self._err
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # SystemExit(0) is a normal exit, not a failure
        if exc_type is SystemExit and not exc.code:
            self.close("ok")
        else:
            self.close("ok" if exc_type is None
                       else f"FAILED: {exc_type.__name__}: {exc}")
        return False


def start_run(name, config=None, root=None):
    return RunDir(name, config=config, root=root)


def latest(name=None, root=None):
    """Most recent run directory, optionally filtered by name prefix."""
    root = Path(root or RESULTS)
    if not root.exists():
        return None
    runs = sorted(d for d in root.iterdir()
                  if d.is_dir() and (name is None or d.name.startswith(name + "_")))
    return runs[-1] if runs else None


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _env():
    env = dict(python=sys.version.split()[0], platform=platform.platform(),
               cwd=os.getcwd())
    for mod in ("numpy", "scipy", "torch", "clawpack"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        env["git"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(CODE),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        env["git"] = None
    return env
