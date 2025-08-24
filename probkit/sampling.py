"""
Random sampling functions for probkit curves and probability functions.

- Exposes a singleton RNG (`rng`) that inherits all of `random.Random` APIs (seed, random, choices, sample, shuffle, etc.)
- Adds probkit helpers on the same object: rng.ntsig, rng.nthsig, rng.biased_curve
- Deterministic control:
    - rng.seed(x): set global sequence for the whole run
    - rng.fork() and rng.spawn(x): create independent RNG instances
    - rng.forked() and rng.spawned(x): context managers yielding independent RNG instances
"""

from random import Random
from typing import Self
from collections.abc import Iterator
from contextlib import contextmanager

from . import curves

__all__ = ["ProbkitRNG", "rng"]


class ProbkitRNG(Random):
    """Singleton-friendly RNG with probkit helpers."""

    # --- probkit helpers ---
    def ntsig(self, k:float)->float:
        """Sample ntsig with random x."""
        return curves.ntsig(k, self.random())

    def nthsig(self, k:float)->float:
        """Sample nthsig with random x."""
        return curves.nthsig(k, self.random())

    def biased_curve(self, k:float, a:float, b:float)->float:
        """Sample biased_curve with random x."""
        return curves.biased_curve(k, a, b, self.random())

    # --- RNG factories ---
    def fork(self) -> Self:
        """Clone the current state into an independent RNG instance."""
        r = type(self)()
        r.setstate(self.getstate())
        return r

    def spawn(self, seed_value:int|float|str|None=None) -> Self:
        """Create an independent seeded RNG instance."""
        r = type(self)()
        r.seed(seed_value)
        return r

    # --- Context helpers which do NOT drift global state ---
    @contextmanager
    def forked(self) -> Iterator[Self]:
        """Context manager yielding a forked RNG instance."""
        yield self.fork()

    @contextmanager
    def spawned(self, seed_value:int|float|str|None=None) -> Iterator[Self]:
        """Context manager yielding a spawned RNG instance."""
        yield self.spawn(seed_value)


# --- Singleton RNG ---
rng = ProbkitRNG()
