"""
Probability modification functions for probkit.
"""

__all__ = ["modified_probability"]

from .utils import effective_ratio

def modified_probability(k:float, a:float|None=None, b:float|None=None, *, ratio:float|None=None) -> float:
    """Modifies a base probability using a ratio.

    Ratio may be provided directly by using the `ratio` keyword argument;
    or by providing both `a` and `b` to calculate the ratio as `a/b`;
    or, if `a` is provided without `b`, then `a` is used as the ratio.

    0 <= base <= 1
    0 <= ratio <= INF

    At ratio=0, modprob=0
    At ratio=1, modprob=base
    At ratio=INF, modprob=1

    https://www.desmos.com/calculator/buzh0oqrxs

    Args:
        k (float): Base probability in [0,1].
        a (float, optional): Numerator of ratio, or ratio itself if `b` is None.
        b (float, optional): Denominator of the ratio.
        ratio (float, keyword-only, optional): Ratio used to modify `k`, specified directly. Overrides `a` and `b`.

    Returns:
        float: Modified probability in [0,1].
    """
    if ratio is not None:
        if a is not None or b is not None:
            raise ValueError("Cannot specify 'a' or 'b' when 'ratio' is provided.")
        x = ratio
    elif a is not None:
        x = a if b is None else effective_ratio(a, b)
    else:
        raise ValueError("Must specify either 'a' (and optionally 'b'), or 'ratio'.")

    if not 0 <= k <= 1:
        raise ValueError(f"Base probability 'k' must be in range [0,1], got {k}.")
    if x < 0:
        raise ValueError(f"Ratio may not be negative, got {x}.")

    if x == 0: return 0
    if x == 1: return k
    if x == float('inf'): return 1
    if k in (0, 1): return k

    kx = k * x
    return kx / (kx - k + 1)
