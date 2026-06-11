"""
A simple timer utility for measuring execution time of code blocks and functions.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable


class Timer:
    """A simple timer class to measure elapsed time."""

    def __init__(self, name: str = ""):
        self.name = name
        self._start: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return the elapsed time."""
        if self._start is None:
            raise RuntimeError("Timer was not started.")
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def __repr__(self) -> str:
        """String representation of the Timer."""
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.3f}s)"


@contextmanager
def timer(name: str = ""):
    """Context manager for timing a block of code with an optional name for identification."""
    t = Timer(name)
    t.start()
    try:
        yield t
    finally:
        t.stop()
        print(f"  [timer] {name or 'block'}: {t.elapsed:.3f}s")


def timed(fn: Callable) -> Callable:
    """Decorator to time a function and print the elapsed time with the function name."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        t = Timer(fn.__name__)
        t.start()
        result = fn(*args, **kwargs)
        t.stop()
        print(f"  [timer] {fn.__name__}: {t.elapsed:.3f}s")
        return result

    return wrapper
