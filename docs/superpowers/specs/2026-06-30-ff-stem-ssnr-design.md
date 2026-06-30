# Design: SSNR display in `apps/ff_stem_ctf.py`

**Date:** 2026-06-30
**Status:** Approved (design); pending spec review
**Author:** brainstorming session

## 1. Goal

Replace the ptychography **PCTF** curve in the marimo app `apps/ff_stem_ctf.py`
with the **spectral signal-to-noise ratio (SSNR)** for direct ptychography, mirroring
the reference implementation `direct_ptychography_ssnr` in scatterem
(`packages/scatterem/scatterem/reconstruction/direct_ptychography.py:2550`).

To keep the ptychography↔ADF comparison meaningful, the ADF-STEM channel is also
converted from a CTF to an SSNR, and a dose control is added so both curves scale
physically.

## 2. Key finding (why this is small)

The app's existing `pctf` cell (`apps/ff_stem_ctf.py:276–299`) **already computes
scatterem's PCTF**. Scatterem forms
`Γ(K,Q) = conj(A(K))·A(K−Q) − A(K)·conj(A(K+Q))`, `PCTF(Q) = ½·Σ_K|Γ|/N_BF`
in `_phase_contrast_transfer_function` — identical to the app's `gamma`/`pctf` loop,
which sweeps a 1-D radial line `Q = nroll·dk` instead of the full 2-D `Q` grid.

(Minor: the app normalizes by `Σ|A|` with a soft-edged aperture, whereas scatterem
divides by the hard bright-field pixel count `2·N_BF`; this is a pre-existing property
of the untouched `pctf` cell and does not affect the noise-model port below.)

Crucially, scatterem's PCTF/SSNR do **not** use the measured data `G` (G only sets
the output array shape). Therefore the SSNR is fully analytic in the aperture
geometry, and the port needs no 4D-STEM data, no scatterem import, and no torch.
This matches the app's style: both `apps/ff_stem_ctf.py` and `apps/ptychographic_ctf.py`
are 100% self-contained numpy notebooks with zero scatterem imports.

**Consequence:** keep the `pctf` cell untouched; only add the noise model that turns
`pctf` into an SSNR.

## 3. Math (pure numpy port)

All quantities in consistent reciprocal-length units (1/Å). Map to existing app
variables: `λ = lam`, `dk = delta_k` (detector reciprocal pixel size),
`scan_step = sampling`, `α = convergence_angle.value·1e-3` (rad).

Frequency axis: define it directly from the curve length,
`q = np.arange(len(pctf)) * dk` (this equals `nroll·dk = kx[:sh[0]//2]` but needs
neither `probe`/`kx` nor `sh`). Use this single definition; see §4 for the marimo
single-definition constraint it must satisfy.

### 3.1 Ptychography SSNR

```
R       = α / λ                      # aperture radius, 1/Å
rBF     = R / dk                     # aperture radius in detector pixels
Nα      = π · rBF²                   # bright-field disk area in pixels
noise²(q) = (N2(q) + N3(q)) / Nα
SSNR_p(q) = fluence · pctf(q)² / noise²(q)
```

`SSNR_p` is set to 0 where `noise² == 0` (q ≥ 2R, apertures no longer overlap).

### 3.2 Overlap pixel counts (exact port of `scatterem/utils/transfer.py:574–637`)

```python
def pair_overlap_area(d, R):
    d = np.asarray(d, dtype=np.float64); A = np.zeros_like(d)
    m = d < 2*R; dm = d[m]
    A[m] = 2*R**2*np.arccos(dm/(2*R)) - 0.5*dm*np.sqrt(4*R**2 - dm**2)
    return A

def triple_overlap_area(q, R):
    q = np.asarray(q, dtype=np.float64); A3 = np.zeros_like(q)
    m = q <= R; qm = q[m]
    A3[m] = np.pi*R**2 - 2*R**2*np.arcsin(qm/R) - 2*qm*np.sqrt(R**2 - qm**2)
    return A3

def double_and_triple_pixel_counts(q, R, delta_k):
    q  = np.asarray(q, dtype=np.float64)
    A3 = triple_overlap_area(q, R)
    A2 = 2*pair_overlap_area(q, R) + pair_overlap_area(2*q, R) - 3*A3
    A2[q >= 2*R] = 0.0
    A3[q >= 2*R] = 0.0
    return A2/delta_k**2, A3/delta_k**2
```

Sanity checks (must hold): at q=0 all three disks coincide → `A2=0`, `N3=Nα`,
`noise=1`, but `pctf(0)=0` (Γ=0) → `SSNR_p(0)=0`. At q≥2R → `noise=0` → `SSNR_p=0`.

### 3.3 ADF SSNR (white / Poisson noise)

```
SSNR_ADF(q) = fluence · η · adf_ctf_base(q)²
```

`adf_ctf_base` is the existing radially-averaged incoherent CTF
(`apps/ff_stem_ctf.py:302–321`), reused unchanged. The previous Lorentzian
`dose_envelope` is **removed** (the dose slider + η now carry the dose dependence;
white noise has no extra q-dependence beyond CTF²).

### 3.4 Dose / fluence

```
fluence = dose · sampling²        # e⁻ per probe position
```

`dose` is the new slider in e⁻/Å². The same `fluence` multiplies **both** channels,
putting them on one dose footing.

### 3.5 Combined SSNR

```
SSNR_combined(q) = SSNR_p(q) + SSNR_ADF(q)
```

Independent channels → inverse variances add → power-SNRs add. This replaces the old
`pctf + adf_ctf` line.

### 3.6 Stated assumptions

Flat object power spectrum, equal per-probe dose for both channels, and
scan-accumulation absorbed into the transfer normalization — the same assumptions
scatterem documents for its analytical SSNR fallback.

## 4. Code changes (4 touch points)

1. **Controls cell** (`:7–64`): remove `adf_suppression`; add
   - `dose` slider in e⁻/Å², logarithmic ~1e2–1e6, default 1e4. `mo.ui.slider` has no
     native log mode in this marimo version, so use an explicit value list, e.g.
     `mo.ui.slider(steps=np.logspace(2, 6, 41).tolist(), value=1e4, ...)`.
   - `adf_efficiency` (η) slider, range 0.01–0.5, default 0.1.
   Update the returned tuple accordingly.

2. **Utilities cell** (the `ComplexProbe` cell, ~`:370`): add the three pure
   functions from §3.2 plus a thin `ptycho_noise_squared(q, R, dk)` helper. These are
   standalone and unit-testable.

3. **New compute cell**: inputs `pctf`, `adf_ctf_base`, `dose`, `adf_efficiency`,
   `sampling`, `dk`, `lam`, `convergence_angle`; outputs `ssnr_ptycho`, `ssnr_adf`,
   `ssnr_combined`, and `q`. Computes `R`, `rBF`, `Nα`,
   `q = np.arange(len(pctf)) * dk`, the noise, and the three SSNRs.

4. **Plot cell** (`fig_pctf`, `:324–360`): plot `ssnr_ptycho` ("ptychography"),
   `ssnr_adf` ("ADF-STEM"), `ssnr_combined` ("ptychography + ADF-STEM"); set
   **log y-axis**, remove `set_ylim(0,1)`, relabel y → "SSNR"; keep the dual
   frequency / real-space-distance x-axes.

   Also update the layout cell's `mo.vstack` (`:248`) and its function signature to
   swap `adf_suppression` → `dose`, `adf_efficiency`.

   **marimo single-definition constraint:** `q` is currently assigned in the
   `fig_pctf` cell (`:331`) and `R` is not yet defined. Because marimo forbids a
   global being assigned in two cells, the new compute cell becomes the sole definer
   of `q` (and `R`); the old `q =` line and the entire `dose_envelope`/`adf_ctf`
   block in `fig_pctf` must be deleted in the same change. Audit any other reused
   names before adding the cell.

## 5. Testing

Pure-numpy unit tests (no scatterem / torch dependency, keeping the app
self-contained):

- `pair_overlap_area`: `d=0 → πR²`; `d≥2R → 0`; monotonically decreasing in d on
  `[0, 2R]`.
- `triple_overlap_area`: `q=0 → πR²`; `q=R → 0`; `q>R → 0`.
- `double_and_triple_pixel_counts`: at q=0 `N2=0` and `N3=Nα`; at q≥2R both 0;
  `N2,N3 ≥ 0` everywhere.
- SSNR assembly: `SSNR_p(0)=0`; `SSNR_p=0` for q≥2R; `SSNR_p,SSNR_ADF ≥ 0`; SSNRs
  scale linearly with `dose` and with `η` (ADF).

A lightweight smoke check that the notebook executes end-to-end with the new cells.

## 6. Out of scope

- 2-D SSNR map (1-D radial only, per decision).
- Changing the ADF base-CTF physics.
- Aberration UI beyond what already exists.
- Numerical cross-check against scatterem's `direct_ptychography_ssnr` (deemed
  overkill for a notebook; the §3.2 port is a line-for-line copy of the reference).
