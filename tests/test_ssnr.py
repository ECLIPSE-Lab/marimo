import numpy as np
import pytest

from ctf.utils import (
    pair_overlap_area,
    triple_overlap_area,
    double_and_triple_pixel_counts,
    ptycho_ssnr,
    adf_ssnr,
)

R = 5.0
DK = 0.1


def test_pair_overlap_zero_separation():
    assert pair_overlap_area(0.0, R) == pytest.approx(np.pi * R**2)


def test_pair_overlap_vanishes_beyond_2R():
    assert pair_overlap_area(2 * R, R) == pytest.approx(0.0)
    assert pair_overlap_area(2.5 * R, R) == pytest.approx(0.0)


def test_pair_overlap_monotonic_decreasing():
    d = np.linspace(0, 2 * R, 50)
    assert np.all(np.diff(pair_overlap_area(d, R)) <= 1e-9)


def test_triple_overlap_zero_separation():
    assert triple_overlap_area(0.0, R) == pytest.approx(np.pi * R**2)


def test_triple_overlap_vanishes_at_and_beyond_R():
    assert triple_overlap_area(R, R) == pytest.approx(0.0)
    assert triple_overlap_area(1.5 * R, R) == pytest.approx(0.0)


def test_pixel_counts_at_dc():
    n2, n3 = double_and_triple_pixel_counts(np.array([0.0]), R, DK)
    assert n2[0] == pytest.approx(0.0)
    assert n3[0] == pytest.approx(np.pi * R**2 / DK**2)  # == Nalpha


def test_pixel_counts_vanish_beyond_2R():
    n2, n3 = double_and_triple_pixel_counts(np.array([2 * R, 3 * R]), R, DK)
    assert np.all(n2 == 0) and np.all(n3 == 0)


def test_pixel_counts_nonnegative():
    q = np.linspace(0, 2.2 * R, 100)
    n2, n3 = double_and_triple_pixel_counts(q, R, DK)
    assert np.all(n2 >= -1e-9) and np.all(n3 >= -1e-9)


def test_ptycho_ssnr_zero_when_pctf_zero():
    q = np.linspace(0, 2.2 * R, 100)
    pctf = np.ones_like(q)
    pctf[0] = 0.0
    assert ptycho_ssnr(pctf, q, R, DK, 1000.0)[0] == 0.0


def test_ptycho_ssnr_zero_beyond_2R():
    q = np.linspace(0, 3 * R, 200)
    ssnr = ptycho_ssnr(np.ones_like(q), q, R, DK, 1000.0)
    assert np.all(ssnr[q >= 2 * R] == 0.0)


def test_ptycho_ssnr_scales_linearly_with_fluence():
    q = np.linspace(0.1 * R, 1.5 * R, 50)
    pctf = np.full_like(q, 0.5)
    s1 = ptycho_ssnr(pctf, q, R, DK, 1.0)
    s2 = ptycho_ssnr(pctf, q, R, DK, 10.0)
    np.testing.assert_allclose(s2, 10.0 * s1)


def test_ptycho_ssnr_analytic_value_at_dc():
    # At q=0: n2=0, n3=Nalpha => noise_sq=1 => SSNR = fluence * pctf^2
    result = ptycho_ssnr(np.array([1.0]), np.array([0.0]), R, DK, 1000.0)
    assert result[0] == pytest.approx(1000.0)


def test_adf_ssnr_formula_and_scaling():
    ctf = np.linspace(0, 1, 20)
    np.testing.assert_allclose(adf_ssnr(ctf, 100.0, 0.1), 100.0 * 0.1 * ctf**2)
    np.testing.assert_allclose(adf_ssnr(ctf, 200.0, 0.1), 2 * adf_ssnr(ctf, 100.0, 0.1))
    np.testing.assert_allclose(adf_ssnr(ctf, 100.0, 0.2), 2 * adf_ssnr(ctf, 100.0, 0.1))


def test_app_inlined_ssnr_matches_reference():
    """The WASM-deployed app inlines its own copy of the SSNR math (sibling modules
    aren't importable in the browser). Guard against drift from this tested reference."""
    import matplotlib

    matplotlib.use("Agg")
    from apps.ff_stem_ssnr import app

    _, defs = app.run()
    q = np.linspace(0, 2.5 * R, 60)
    pctf = np.linspace(0, 1, 60)
    np.testing.assert_allclose(
        defs["ptycho_ssnr"](pctf, q, R, DK, 500.0), ptycho_ssnr(pctf, q, R, DK, 500.0)
    )
    np.testing.assert_allclose(
        defs["adf_ssnr"](pctf, 500.0, 0.3), adf_ssnr(pctf, 500.0, 0.3)
    )
