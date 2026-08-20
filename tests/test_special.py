"""Special functions pinned against published critical values.

These are the numerical foundation of stats/pvalue-mismatch. If they drift,
every p-value finding drifts with them, so they are pinned to values from
standard tables rather than to whatever the implementation happens to return.
"""

import math

import pytest

from resint.mathx.special import betainc, chi2_sf, f_sf, gammainc_upper, norm_sf, t_sf

TOL = 5e-5


@pytest.mark.parametrize(
    "t, df, expected",
    [
        (2.086, 20, 0.05),      # two-tailed .05 critical value
        (2.228, 10, 0.05),
        (1.0, 10, 0.34086),
        (0.0, 5, 1.0),
        (3.169, 10, 0.01),
    ],
)
def test_t_two_tailed(t, df, expected):
    assert 2 * t_sf(t, df) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize(
    "x, df, expected",
    [
        (3.8415, 1, 0.05),
        (5.9915, 2, 0.05),
        (10.0, 5, 0.07524),
        (0.0, 3, 1.0),
    ],
)
def test_chi2(x, df, expected):
    assert chi2_sf(x, df) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize(
    "f, df1, df2, expected",
    [
        (4.3512, 1, 20, 0.05),
        (3.4928, 2, 20, 0.05),
        (1.0, 5, 5, 0.5),
    ],
)
def test_f(f, df1, df2, expected):
    assert f_sf(f, df1, df2) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize(
    "z, expected",
    [(1.959964, 0.025), (0.0, 0.5), (2.575829, 0.005)],
)
def test_normal(z, expected):
    assert norm_sf(z) == pytest.approx(expected, abs=TOL)


def test_t_symmetry():
    assert t_sf(-1.5, 12) == pytest.approx(1 - t_sf(1.5, 12), abs=1e-12)


def test_betainc_endpoints_and_symmetry():
    assert betainc(2, 3, 0.0) == 0.0
    assert betainc(2, 3, 1.0) == 1.0
    assert betainc(3, 3, 0.5) == pytest.approx(0.5, abs=1e-12)
    # I_x(a,b) == 1 - I_{1-x}(b,a)
    assert betainc(2.5, 4.5, 0.3) == pytest.approx(1 - betainc(4.5, 2.5, 0.7), abs=1e-12)


def test_gammainc_upper_bounds():
    assert gammainc_upper(2.0, 0.0) == 1.0
    assert gammainc_upper(1.0, 1.0) == pytest.approx(math.exp(-1.0), abs=1e-12)


@pytest.mark.parametrize("bad", [(0, 2, 0.5), (2, 0, 0.5)])
def test_betainc_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        betainc(*bad)


def test_betainc_rejects_x_outside_unit_interval():
    with pytest.raises(ValueError):
        betainc(2, 2, 1.5)


@pytest.mark.parametrize("fn", [t_sf, chi2_sf])
def test_rejects_nonpositive_df(fn):
    with pytest.raises(ValueError):
        fn(1.0, 0)
