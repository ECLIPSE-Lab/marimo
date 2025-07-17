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