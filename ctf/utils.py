import math as ma
import numpy as np

def electron_wavelength_angstrom(E_eV):
    """ returns relativistic electron wavelength in Angstroms. """
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34

    lam = h / ma.sqrt(2 * m * e * E_eV) / ma.sqrt(1 + e * E_eV / 2 / m / c**2) *1e10
    return lam

def compute_ctf(phase):
    """ returns the 2D FFT amplitude (CTF) of input phase, normalizing the DC frequency. """
    ctf = np.abs(np.fft.fft2(phase))
    # crude DC estimation
    ctf[0,0] = ctf[[-1,-1,-1,0,0,1,1,1],[-1,0,1,-1,1,-1,0,1]].mean()
    return ctf

def radially_average_ctf(corner_centered_ctf,sampling):
    """ returns the radially-averaged CTF of a corner-centered 2D CTF array. """
    nx, ny = corner_centered_ctf.shape
    sx, sy = sampling
    
    kx = np.fft.fftfreq(nx,sx)
    ky = np.fft.fftfreq(ny,sy)
    k  = np.sqrt(kx[:,None]**2 + ky[None,:]**2).ravel()

    intensity = corner_centered_ctf.ravel()

    bin_size = kx[1]-kx[0]
    k_bins = np.arange(0, k.max() + bin_size, bin_size)

    inds = k / bin_size
    inds_f = np.floor(inds).astype("int")
    d_ind = inds - inds_f

    nf = np.bincount(inds_f, weights=(1 - d_ind), minlength=k_bins.shape[0])
    nc = np.bincount(inds_f + 1, weights=(d_ind), minlength=k_bins.shape[0])
    n = nf + nc

    I_bins0 = np.bincount(
        inds_f, weights=intensity * (1 - d_ind), minlength=k_bins.shape[0]
    )
    I_bins1 = np.bincount(
        inds_f + 1, weights=intensity * (d_ind), minlength=k_bins.shape[0]
    )

    I_bins = (I_bins0 + I_bins1) / n

    inds = k_bins <= np.abs(kx).max()

    return k_bins[inds], I_bins[inds]

def return_patch_indices(positions_px,roi_shape,obj_shape):
    """ """
    x0 = np.round(positions_px[:, 0]).astype("int")
    y0 = np.round(positions_px[:, 1]).astype("int")

    x_ind = np.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
    y_ind = np.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

    row = (x0[:, None, None] + x_ind[None, :, None]) % obj_shape[0]
    col = (y0[:, None, None] + y_ind[None, None, :]) % obj_shape[1]

    return row, col

def sum_patches_base(patches, positions_px, roi_shape, object_shape):
    """ """
    
    x0 = np.round(positions_px[:, 0]).astype("int")
    y0 = np.round(positions_px[:, 1]).astype("int")

    x_ind = np.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
    y_ind = np.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

    flat_weights = patches.ravel()
    indices = ((y0[:, None, None] + y_ind[None, None, :]) % object_shape[1]) + (
        (x0[:, None, None] + x_ind[None, :, None]) % object_shape[0]
    ) * object_shape[1]
    
    counts = np.bincount(
        indices.ravel(), weights=flat_weights, minlength=np.prod(object_shape)
    )
    counts = np.reshape(counts, object_shape)
    return counts

def sum_patches(patches, positions_px, roi_shape, obj_shape):
    """ """

    if np.iscomplexobj(patches):
        real = sum_patches_base(
            patches.real, positions_px, roi_shape, obj_shape
        )
        imag = sum_patches_base(
            patches.imag, positions_px, roi_shape, obj_shape
        )
        return real + 1.0j * imag
    else:
        return sum_patches_base(patches, positions_px, roi_shape, obj_shape)

def simulate_data(complex_obj, probe_array, row, col):
    """ """
    arr = np.asarray(complex_obj,dtype=np.complex128)
    probe = np.asarray(probe_array,dtype=np.complex128)

    obj_patches = arr[row,col]
    exit_waves = obj_patches*probe
    amplitudes = np.abs(np.fft.fft2(exit_waves))
    return amplitudes


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
    q = np.atleast_1d(q)
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
    Returns 0 where the apertures no longer overlap (``q >= 2R``); there the N2/N3
    pixel counts are clamped to 0, so there is no signal to recover.

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
