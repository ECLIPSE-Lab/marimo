# FF-STEM Ptychography + ADF SSNR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ptychography PCTF curve in `apps/ff_stem_ctf.py` with a spectral SNR (SSNR), and put the ADF-STEM channel on the same SSNR footing, driven by a new electron-dose control.

**Architecture:** The app's existing `pctf` cell already computes scatterem's analytical PCTF (a 1-D radial trotter-overlap transfer function). We add a small pure-numpy noise model — a line-for-line port of scatterem's circle-overlap math — to `ctf/utils.py`, turning that PCTF into an SSNR, and add an analogous white-noise SSNR for ADF. The notebook imports these helpers and a new compute cell + rewritten plot cell display three SSNR curves (ptychography, ADF, combined) on a log axis. No scatterem/torch dependency.

**Tech Stack:** Python, numpy, marimo 0.23.11, matplotlib, pytest 8.4.2.

---

## Background facts (verified)

- `ctf/utils.py` is the repo's pure-numpy helper module, re-exported via `ctf/__init__.py`. A byte-identical mirror exists at `apps/ctf/` so `from ctf.utils import …` resolves whether the app runs from repo root **or** from `apps/`. **`ctf/` is the source of truth; `apps/ctf/` must be kept identical.**
- `mo.ui.slider` in this marimo version supports `steps=<list>` (confirmed via signature); `np.logspace(2, 6, 41)` includes exactly `1e4`.
- Headless smoke test that works today: from repo root,
  `MPLBACKEND=Agg python3 -c "import matplotlib; matplotlib.use('Agg'); from apps.ff_stem_ctf import app; app.run()"` → returns a tuple, no exception.
- pytest must be run as `python -m pytest …` from the repo root so the repo root is on `sys.path` and `import ctf.utils` resolves.

## Reference math (the contract the port must satisfy)

For radial spatial frequency `q` (1/Å), aperture radius `R = α/λ` (1/Å), detector reciprocal pixel `delta_k` (1/Å):

- Ptychography: `SSNR_p(q) = fluence · PCTF(q)² / noise(q)²`, `noise² = (N2+N3)/Nα`, `Nα = π·(R/delta_k)²`. Zero where `q ≥ 2R`.
- N2/N3 from circle overlaps (scatterem `utils/transfer.py:574–637`): `pair(d,R)=2R²arccos(d/2R) − ½d√(4R²−d²)` (0 for d≥2R); `A3=triple(q,R)=πR² − 2R²arcsin(q/R) − 2q√(R²−q²)` (0 for q>R); `A2 = 2·pair(q,R) + pair(2q,R) − 3·A3`; both →0 for q≥2R; `N2=A2/delta_k²`, `N3=A3/delta_k²`.
- ADF: `SSNR_ADF(q) = fluence · η · CTF_ADF(q)²` (white/Poisson noise).
- `fluence = dose · scan_step²` (e⁻/probe); `dose` in e⁻/Å²; combined `= SSNR_p + SSNR_ADF`.

Spec: `docs/superpowers/specs/2026-06-30-ff-stem-ssnr-design.md`.

---

## File Structure

- **Modify** `ctf/utils.py` — add 5 pure functions: `pair_overlap_area`, `triple_overlap_area`, `double_and_triple_pixel_counts`, `ptycho_ssnr`, `adf_ssnr`.
- **Modify** `ctf/__init__.py` — re-export the 5 functions.
- **Sync** `apps/ctf/utils.py`, `apps/ctf/__init__.py` — copy from `ctf/` (keep byte-identical).
- **Create** `tests/test_ssnr.py` — pytest unit tests for the 5 functions.
- **Modify** `apps/ff_stem_ctf.py` — controls cell (swap slider), imports cell (import helpers), new compute cell, plot cell (rewrite), layout cell (signature + vstack).

---

## Task 1: SSNR math in `ctf/utils.py` (TDD)

**Files:**
- Create: `tests/test_ssnr.py`
- Modify: `ctf/utils.py`, `ctf/__init__.py`
- Sync: `apps/ctf/utils.py`, `apps/ctf/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ssnr.py`:

```python
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


def test_ptycho_ssnr_dc_is_zero():
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


def test_adf_ssnr_formula_and_scaling():
    ctf = np.linspace(0, 1, 20)
    np.testing.assert_allclose(adf_ssnr(ctf, 100.0, 0.1), 100.0 * 0.1 * ctf**2)
    np.testing.assert_allclose(adf_ssnr(ctf, 200.0, 0.1), 2 * adf_ssnr(ctf, 100.0, 0.1))
    np.testing.assert_allclose(adf_ssnr(ctf, 100.0, 0.2), 2 * adf_ssnr(ctf, 100.0, 0.1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from repo root): `python -m pytest tests/test_ssnr.py -q`
Expected: collection succeeds, all tests FAIL with `ImportError: cannot import name 'pair_overlap_area' from 'ctf.utils'`.
(If instead you get `ModuleNotFoundError: No module named 'ctf'`, you are not in the repo root — `cd` there.)

- [ ] **Step 3: Implement the 5 functions in `ctf/utils.py`**

Append to `ctf/utils.py` (it already has `import numpy as np` at the top):

```python


def pair_overlap_area(d, R):
    """Overlap area of two circles of radius ``R`` with centre separation ``d``.

    Returns 0 where ``d >= 2R``. Vectorised over ``d``. Port of scatterem
    ``utils/transfer.py:pair_overlap_area``.
    """
    d = np.asarray(d, dtype=np.float64)
    area = np.zeros_like(d)
    mask = d < 2 * R
    dm = d[mask]
    area[mask] = 2 * R**2 * np.arccos(dm / (2 * R)) - 0.5 * dm * np.sqrt(4 * R**2 - dm**2)
    return area


def triple_overlap_area(q, R):
    """Triple-overlap area of three radius-``R`` circles centred at -q, 0, +q.

    Nonzero only for ``0 <= q <= R``. Port of scatterem
    ``utils/transfer.py:triple_overlap_area``.
    """
    q = np.asarray(q, dtype=np.float64)
    area = np.zeros_like(q)
    mask = q <= R
    qm = q[mask]
    area[mask] = np.pi * R**2 - 2 * R**2 * np.arcsin(qm / R) - 2 * qm * np.sqrt(R**2 - qm**2)
    return area


def double_and_triple_pixel_counts(q, R, delta_k):
    """Detector-pixel counts in the double- (N2) and triple- (N3) overlap regions.

    Three radius-``R`` apertures centred at -q, 0, +q; ``delta_k`` is the detector
    reciprocal-pixel size. q, R, delta_k must share one reciprocal-length unit
    (1/Angstrom). Both counts are 0 for ``q >= 2R``. Port of scatterem
    ``utils/transfer.py:double_and_triple_pixel_counts``.
    """
    q = np.asarray(q, dtype=np.float64)
    a3 = triple_overlap_area(q, R)
    a2 = 2 * pair_overlap_area(q, R) + pair_overlap_area(2 * q, R) - 3 * a3
    a2[q >= 2 * R] = 0.0
    a3[q >= 2 * R] = 0.0
    return a2 / delta_k**2, a3 / delta_k**2


def ptycho_ssnr(pctf, q, R, delta_k, fluence):
    """Analytical direct-ptychography SSNR from a 1-D radial PCTF.

    Mirrors scatterem ``direct_ptychography_ssnr``:
    ``SSNR(q) = fluence * PCTF(q)**2 / noise(q)**2`` with
    ``noise**2 = (N2 + N3) / Nalpha`` and ``Nalpha = pi * (R / delta_k)**2``.
    Returns 0 where the apertures no longer overlap (``q >= 2R`` -> noise 0).

    pctf, q : 1-D arrays of equal length. R, delta_k : 1/Angstrom.
    fluence : electrons per probe position.
    """
    pctf = np.asarray(pctf, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n2, n3 = double_and_triple_pixel_counts(q, R, delta_k)
    n_alpha = np.pi * (R / delta_k) ** 2
    noise_sq = (n2 + n3) / n_alpha
    ssnr = np.zeros_like(pctf)
    nz = noise_sq > 0
    ssnr[nz] = fluence * pctf[nz] ** 2 / noise_sq[nz]
    return ssnr


def adf_ssnr(adf_ctf, fluence, efficiency):
    """Incoherent ADF-STEM SSNR under white (Poisson) noise.

    ``SSNR(q) = fluence * efficiency * CTF(q)**2``. fluence is electrons per probe
    position; efficiency is the fraction of electrons reaching the annular detector.
    """
    adf_ctf = np.asarray(adf_ctf, dtype=np.float64)
    return fluence * efficiency * adf_ctf**2
```

- [ ] **Step 4: Re-export from `ctf/__init__.py`**

Edit the second line of `ctf/__init__.py` to append the new names:

```python
from ctf.utils import electron_wavelength_angstrom, compute_ctf, radially_average_ctf, return_patch_indices, simulate_data, sum_patches, pair_overlap_area, triple_overlap_area, double_and_triple_pixel_counts, ptycho_ssnr, adf_ssnr
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_ssnr.py -q`
Expected: `12 passed`.

- [ ] **Step 6: Sync the `apps/ctf/` mirror**

Run (from repo root): `cp ctf/utils.py apps/ctf/utils.py && cp ctf/__init__.py apps/ctf/__init__.py`
Verify identical: `diff ctf/utils.py apps/ctf/utils.py && diff ctf/__init__.py apps/ctf/__init__.py && echo IN_SYNC`
Expected: `IN_SYNC`.

- [ ] **Step 7: Commit**

```bash
git add ctf/utils.py ctf/__init__.py apps/ctf/utils.py apps/ctf/__init__.py tests/test_ssnr.py
git commit -m "feat(ctf): add ptychography + ADF SSNR math with tests"
```

---

## Task 2: Wire SSNR into `apps/ff_stem_ctf.py`

These edits leave the notebook DAG inconsistent mid-task (a cell may reference a name another cell hasn't been updated to define yet). That is expected — do **all** edits, then run the smoke test once at the end, then commit. Do not commit a half-edited notebook.

**Files:** Modify `apps/ff_stem_ctf.py`

- [ ] **Step 1: Controls cell — replace the `adf_suppression` slider with `dose` + `adf_efficiency`**

The controls cell signature is `def _(mo):` (line ~8). Change it to `def _(mo, np):`.

Replace this block (lines ~39–41):

```python
    adf_suppression = mo.ui.slider(
        start=1, stop=50, value=10, step=1, label="ADF dose falloff γ [Å]", show_value=True
    )
```

with:

```python
    dose = mo.ui.slider(
        steps=np.logspace(2, 6, 41).tolist(),
        value=1e4,
        label="dose [e⁻/Å²]",
        show_value=True,
    )

    adf_efficiency = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.1,
        label="ADF efficiency η", show_value=True,
    )
```

Replace the controls cell's return block (lines ~56–64):

```python
    return (
        adf_suppression,
        astigmatism,
        astigmatism_angle_slider,
        convergence_angle,
        detector,
        energy,
        max_semiangle,
    )
```

with:

```python
    return (
        adf_efficiency,
        astigmatism,
        astigmatism_angle_slider,
        convergence_angle,
        detector,
        dose,
        energy,
        max_semiangle,
    )
```

- [ ] **Step 2: Imports cell — import the SSNR helpers**

In the imports cell (lines ~698–705), add after the `colorspacious` import:

```python
    from ctf.utils import ptycho_ssnr, adf_ssnr
```

and change its return line from

```python
    return AnchoredSizeBar, cspace_convert, mo, np, plt
```

to

```python
    return AnchoredSizeBar, adf_ssnr, cspace_convert, mo, np, plt, ptycho_ssnr
```

- [ ] **Step 3: Add a new compute cell**

Insert this cell immediately **after** the `adf_ctf_base` cell (the one returning `(adf_ctf_base,)`, ~line 321) and **before** the plot cell:

```python
@app.cell
def _(
    adf_ctf_base,
    adf_efficiency,
    adf_ssnr,
    convergence_angle,
    dk,
    dose,
    lam,
    np,
    pctf,
    ptycho_ssnr,
    sampling,
):
    # Spatial-frequency axis for the 1-D radial PCTF: q[n] = n * dk  (1/Å).
    q = np.arange(len(pctf)) * dk
    R = (convergence_angle.value * 1e-3) / lam          # aperture radius, 1/Å
    fluence = dose.value * sampling**2                   # e⁻ per probe position

    ssnr_ptycho = ptycho_ssnr(pctf, q, R, dk, fluence)
    ssnr_adf = adf_ssnr(adf_ctf_base, fluence, adf_efficiency.value)
    ssnr_combined = ssnr_ptycho + ssnr_adf
    return R, fluence, q, ssnr_adf, ssnr_combined, ssnr_ptycho
```

- [ ] **Step 4: Rewrite the plot cell**

Replace the entire plot cell (lines ~324–360, `def _(adf_ctf_base, adf_suppression, np, pctf, plt, probe, sh):` … `return (fig_pctf,)`) with:

```python
@app.cell
def _(np, plt, probe, sh, ssnr_adf, ssnr_combined, ssnr_ptycho):
    kx, ky = probe.get_spatial_frequencies()

    # SSNR is 0 at DC and beyond the 2R aperture-overlap cutoff; mask those to NaN
    # so the log plot shows clean gaps instead of log(0) warnings.
    def _pos(a):
        a = np.asarray(a, dtype=float).copy()
        a[a <= 0] = np.nan
        return a

    fig_pctf, ax_pctf = plt.subplots(figsize=(16.8 / 1.5, 3))
    ax_pctf.semilogy(_pos(ssnr_ptycho), label='ptychography')
    ax_pctf.semilogy(_pos(ssnr_adf), label='ADF-STEM')
    ax_pctf.semilogy(_pos(ssnr_combined), label='ptychography + ADF-STEM')
    ax_pctf.legend()
    ax_pctf.set_ylabel('SSNR')
    ax_pctf.set_xlabel('spatial frequency [1/Å]')
    ax_pctf.set_xlim(0, sh[0] / 2)

    xtick_positions = np.linspace(0, sh[0] / 2, 16)
    ax_pctf.set_xticks(xtick_positions)
    w = kx[: sh[0] // 2][:: sh[0] // 32]
    ax_pctf.set_xticklabels([f'{x:.1f}' for x in w])

    ax_top = ax_pctf.twiny()
    w = kx[: sh[0] // 2][:: sh[0] // 32]
    w = w[1:]
    ax_top.set_xticks(xtick_positions[1:])
    ax_top.set_xticklabels([f'{1 / x:.1f}' for x in w])
    ax_top.set_xlabel('spatial distances [Å]')
    plt.show()
    return (fig_pctf,)
```

- [ ] **Step 5: Update the layout cell (signature + vstack)**

In the layout cell (the long `def _(` starting ~line 122), in its parameter list:
- remove the `    adf_suppression,` line,
- add `    adf_efficiency,` and `    dose,` lines (anywhere in the list; keep it tidy).

Then in that cell, replace the `vertical` vstack list (line ~248):

```python
        [energy,max_semiangle,detector,convergence_angle,defocus, adf_suppression, text_nyquist,text7,text1, text2, text3, text4],
```

with:

```python
        [energy,max_semiangle,detector,convergence_angle,defocus, dose, adf_efficiency, text_nyquist,text7,text1, text2, text3, text4],
```

- [ ] **Step 6: Confirm reference swap is complete**

Run: `grep -n adf_suppression apps/ff_stem_ctf.py`
Expected: no output (exit status 1) — no stale references remain.

Run: `grep -c "adf_efficiency" apps/ff_stem_ctf.py; grep -c "dose" apps/ff_stem_ctf.py`
Expected: each ≥ 3 (definition + controls return + layout signature + vstack [+ compute cell]) — confirms the new names were actually added, not just the old one removed.

- [ ] **Step 7: Headless smoke test**

Run (from repo root):

```bash
MPLBACKEND=Agg python3 -c "import matplotlib; matplotlib.use('Agg'); from apps.ff_stem_ctf import app; app.run()"
```

Expected: completes with no exception (prints the probe metrics block as before). A `marimo` DAG error here means a cell references a name that is not defined/returned — re-check Steps 1–5 (common cause: a name missing from a cell's `return`, or a duplicate definition of `q`/`R`).

- [ ] **Step 8: Commit**

```bash
git add apps/ff_stem_ctf.py
git commit -m "feat(ff_stem_ctf): show ptychography + ADF SSNR with dose control"
```

---

## Task 3: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests/ -q`
Expected: `12 passed`.

- [ ] **Step 2: Mirror integrity**

Run: `diff ctf/utils.py apps/ctf/utils.py && diff ctf/__init__.py apps/ctf/__init__.py && echo IN_SYNC`
Expected: `IN_SYNC`.

- [ ] **Step 3: Headless app run (both invocation dirs)**

```bash
MPLBACKEND=Agg python3 -c "import matplotlib; matplotlib.use('Agg'); from apps.ff_stem_ctf import app; app.run()"
cd apps && MPLBACKEND=Agg python3 -c "import matplotlib; matplotlib.use('Agg'); import ff_stem_ctf; ff_stem_ctf.app.run()"; cd ..
```

Expected: both complete with no exception (proves `from ctf.utils import …` resolves from repo root and from `apps/`).

- [ ] **Step 4: Manual visual check (optional but recommended)**

Run: `marimo edit apps/ff_stem_ctf.py` and confirm: the lower panel shows three SSNR curves on a log y-axis labelled "SSNR"; moving the **dose** slider scales all curves; moving **ADF efficiency η** scales only the ADF (and combined) curve; the old "ADF dose falloff γ" slider is gone.

---

## Notes for the implementer

- **DRY/source of truth:** only edit `ctf/`; regenerate `apps/ctf/` by copy. Never hand-edit `apps/ctf/`.
- **marimo single-definition rule:** every global is assigned in exactly one cell. `q`, `R`, `fluence`, `ssnr_*` live only in the new compute cell; ensure the old plot-cell `q = kx[...]` line is gone (it is, since Step 4 replaces the whole cell).
- **No scatterem/torch import** anywhere — the port is self-contained numpy by design.
