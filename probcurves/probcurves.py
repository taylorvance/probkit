import random

def modified_probability(k=0.5, a=None, b=None, *, ratio=None):
    """Modifies a base probability using a ratio.

    Users can specify the ratio directly by using the 'ratio' keyword argument,
    or they can provide 'a' and 'b' to calculate the ratio as 'a/b',
    or if 'a' is provided without 'b', then 'a' is used as the ratio.

    0 <= base <= 1
    0 <= ratio <= INF

    At ratio=0, modprob=0
    At ratio=1, modprob=base
    At ratio=INF, modprob=1

    https://www.desmos.com/calculator/buzh0oqrxs

    Args:
        k (float): The base probability, in range [0,1], to be modified.
        a (float, optional): The numerator of the ratio, or the ratio itself if 'b' is None.
        b (float, optional): The denominator of the ratio. Ignored if 'ratio' is provided.
        ratio (float, keyword-only, optional): The ratio used to modify 'k', specified directly. Overrides 'a' and 'b'.

    Returns:
        float: The modified probability in range [0,1].
    """
    # Calculate the ratio (x).
    if ratio is not None:
        if a is not None or b is not None:
            raise ValueError("Cannot specify 'a' or 'b' when 'ratio' is provided.")
        x = ratio
    elif a is not None:
        x = a if b is None else effective_ratio(a,b)
    else:
        raise ValueError("Must specify either 'a' (and optionally 'b'), or 'ratio'.")

    if k < 0 or k > 1:
        raise ValueError("Base probability 'k' must be in range [0,1].", k)
    if x < 0:
        raise ValueError("Ratio may not be negative.", x)

    if x == 0: return 0
    if x == 1: return k
    if x == float('inf'): return 1

    if k in (0,1): return k

    # return 2 * math.atan(x * math.tan(k * math.pi / 2)) / math.pi
    return k*x / (k*x - k + 1)

def _dd_ntsig(k, x=None):
    """Dino Dini's Normalized Tunable Sigmoid

    Calculates a point along a sigmoid-like curve that goes through (-1,-1) and (0,0) and (1,1).
    Based on Dino Dini's normalized tunable sigmoid function: https://dinodini.wordpress.com/2010/04/05/normalized-tunable-sigmoid-functions/

    Negative k is steep (sigmoid curve), bringing all values of y toward -1 (for x<0) or 1 (for x>0).
    Neutral k is linear between (-1,-1) and (1,1).
    Positive k is flat (logit/log-odds curve), bringing all values of y toward 0.

    If x is not provided, uses a random x in range [-1,1].

    https://www.desmos.com/calculator/hvgetz6trt

    Args:
        k (float): Must be in range [-1,1]. The shape (flatness) of the curve.
        x (float): Must be in range [-1,1]. The function will return the corresponding y value at x.
    Returns:
        float: The y value at x.
    """
    if x is None: x = random.uniform(-1, 1)

    if k < -1 or k > 1: raise ValueError("Argument k must be in range [-1,1].", k)
    if x < -1 or x > 1: raise ValueError("Argument x must be in range [-1,1].", x)

    if x in [-1,0,1]: return x

    if k == 1: return 0
    if k == 0: return x
    if k == -1: return -1 if x<0 else 1

    return (k-1)*x / (2*k*abs(x) - k - 1)

def ntsig(k, x=None):
    """Normalized Tunable Sigmoid

    Calculates a point along a sigmoid-like curve that goes through (0,0) and (0.5,0.5) and (1,1).

    Negative k is flat (logit/log-odds curve), bringing all values of y toward 0.5.
    Neutral k is linear between (0,0) and (1,1).
    Positive k is steep/swingy (sigmoid curve), bringing all values of y toward either 0 (for x<0.5) or 1 (for x>0.5).

    If x is not provided, uses a random x in range [0,1).

    https://www.desmos.com/calculator/2hkedvscyv

    Args:
        k (float): Must be in range [-1,1]. The shape (steepness/swinginess) of the curve.
        x (float): Must be in range [0,1]. The function will return the corresponding y value at x.
    Returns:
        float: The y value at x.
    """
    if x is None: x = random.random()

    if k < -1 or k > 1: raise ValueError("Argument 'k' must be in range [-1,1].", k)
    if x < 0 or x > 1: raise ValueError("Argument 'x' must be in range [0,1].", x)

    if x in [0,0.5,1]: return x

    if k == -1: return 0.5
    if k == 0: return x
    if k == 1: return 0 if x<0.5 else 1

    # Translate Dino's nts by [1,1], scale by half, and flip the sign of k (personal preference).
    x = transform_range(x, (0,1), (-1,1))
    return (_dd_ntsig(-k,x) + 1) / 2

def nthsig(k, x=None):
    """Normalized Tunable Half-Sigmoid

    Calculates a point along a half-sigmoid-like curve between (0,0) and (1,1).

    Negative k is convex, bringing all values of y toward 0.
    Neutral k is linear between (0,0) and (1,1).
    Positive k is concave, bringing all values of y toward 1.

    If x is not provided, uses a random x in range [0,1).

    https://www.desmos.com/calculator/2hkedvscyv

    Args:
        k (float): Must be in range [-1,1]. The shape of the curve.
        x (float): Must be in range [0,1]. The function will return the corresponding y value at x.
    Returns:
        float: The y value at x.
    """
    if x is None: x = random.random()

    if k < -1 or k > 1: raise ValueError("Argument 'k' must be in range [-1,1].", k)
    if x < 0 or x > 1: raise ValueError("Argument 'x' must be in range [0,1].", x)

    if x in [0,1]: return x

    if k == -1: return 0
    if k == 0: return x
    if k == 1: return 1

    # Flip the sign of k (personal preference) for Dino's function.
    return _dd_ntsig(-k, x)

def biased_curve(k=0, a=0, b=1, x=None):
    """Calculates a point along a curve between (0,a) and (1,b). The bias parameter (k) bends the curve toward either a or b.

    Negative k favors a, bending the knee of the curve toward (1,a).
    Neutral k is unbiased, linear between (0,a) and (1,b).
    Positive k favors b, bending the knee of the curve toward (0,b).

    If x is not provided, uses a random x in range [0,1).

    https://www.desmos.com/calculator/2hkedvscyv

    Args:
        a (float): The y value at x=0.
        b (float): The y value x=1.
        k (float): Changes the shape of the curve so that all y values move toward either a (when k is negative), b (when k is positive), or neither (when k is 0).
        x (float): Must be in range [0,1]. The function will return the corresponding y value at x.
    Returns:
        float: The y value at x.
    """
    return a + (b-a) * nthsig(k, x)

def transform_range(x, oldrange, newrange=(0,1)):
    """Translates and/or scales a variable from one range to another.

    If newrange is not provided, normalize x on (0,1).

    Args:
        x (float): The value of x relative to oldrange.
        oldrange (tuple): The (min,max) values of the old range. Note: min does not necessarily have to be <max.
        newrange (tuple): The (min,max) values of the new range. Note: min does not necessarily have to be <max.
    Returns:
        float: The transformed value of x relative to newrange.
    """
    return (x-oldrange[0]) * (newrange[1]-newrange[0]) / (oldrange[1]-oldrange[0]) + newrange[0]

def clamp(val, min_val, max_val):
    return min(max(val, min_val), max_val)

def effective_ratio(a, b):
    if a == 0: return 0
    if b == 0: return float('inf') if a > 0 else float('-inf')
    return a / b
