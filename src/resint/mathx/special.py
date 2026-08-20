"""Special functions, implemented locally so install stays light.

A linter's first impression is how fast it starts. Pulling scipy in for four
CDFs would triple install size for a tool whose identity is "instant, no
setup", so the regularized incomplete beta and gamma functions live here.
Both are standard continued-fraction / series expansions and are pinned by
tests against published critical values.
"""

from __future__ import annotations

import math

_EPS = 3.0e-16
_FPMIN = 1.0e-300
_MAX_ITER = 300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h

    raise ArithmeticError(f"betacf failed to converge for a={a}, b={b}, x={x}")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must lie in [0, 1], got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _gamma_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a, x) by series expansion."""
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    raise ArithmeticError(f"gamma series failed to converge for a={a}, x={x}")


def _gamma_cf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a, x) by continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    raise ArithmeticError(f"gamma continued fraction failed for a={a}, x={x}")


def gammainc_upper(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError(f"require x >= 0 and a > 0, got a={a}, x={x}")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


# --- survival functions -------------------------------------------------


def norm_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    half = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return half if t >= 0 else 1.0 - half


def chi2_sf(x: float, df: float) -> float:
    """P(X > x) for chi-squared with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if x <= 0:
        return 1.0
    return gammainc_upper(df / 2.0, x / 2.0)


def f_sf(f: float, df1: float, df2: float) -> float:
    """P(F > f) for an F distribution with (df1, df2) degrees of freedom."""
    if df1 <= 0 or df2 <= 0:
        raise ValueError(f"df must be positive, got df1={df1}, df2={df2}")
    if f <= 0:
        return 1.0
    return betainc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))
