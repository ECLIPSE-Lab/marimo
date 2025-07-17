import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from colorspacious import cspace_convert


def histogram_scaling(array, vmin=0.02, vmax=0.98, normalize=True):
    """ Scales array by clipping values outside the `vmin` and `vmax` quantiles. """
    scaled_array = array.copy()

    if np.min(scaled_array) == 0.0 and np.max(scaled_array) == 0.0:
        return scaled_array
    elif np.isclose(np.max(scaled_array), np.min(scaled_array)):
        if normalize:
            scaled_array /= scaled_array.max()
        return scaled_array     
    else:
        vals = np.sort(scaled_array[~np.isnan(scaled_array)])

        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin = vals[ind_vmin]
        vmax = vals[ind_vmax]

        scaled_array = np.where(scaled_array < vmin, vmin, scaled_array)
        scaled_array = np.where(scaled_array > vmax, vmax, scaled_array)
    
        if normalize:
            scaled_array -= scaled_array.min()
            scaled_array /= scaled_array.max()
    
        return scaled_array


def complex_to_rgb(array, vmin=0.02, vmax=0.98):
    """ Converts complex-valued array to a perceptually-uniform RGB array. """
    amp = histogram_scaling(
        np.abs(array), 
        vmin=vmin, 
        vmax=vmax, 
        normalize=True
    ).clip(1e-16, 1)
    phase = np.angle(array)

    J = amp * 61.5  # restrict luminance to the monotonic chroma cutoff
    C = np.minimum(98 * J / 123, 110)
    h = np.rad2deg(phase) + 180

    JCh = np.stack((J, C, h), axis=-1)
    rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

    return rgb


def combined_images_rgb(list_of_arrays, vmin=0.02, vmax=0.98):
    """ Converts a list of arrays to a perceptually-uniform RGB array. """
    bins = np.asarray(list_of_arrays)
    n = bins.shape[0]
    hue_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    complex_arr = np.tensordot(np.exp(1j * hue_angles), bins, axes=1)

    return complex_to_rgb(complex_arr, vmin=vmin, vmax=vmax)

def add_scalebar(ax, length, sampling, units, size_vertical=1, pad=0.5, color="white"):
    """ """
    if abs(1.0 - length*sampling) <= 1e-2:
        txt = units
    else:
        txt = f"{sampling*length:.0f} {units}"
        
    bar = AnchoredSizeBar(
        ax.transData,
        length,
        txt,
        "lower right",
        pad=pad,
        color=color,
        frameon=False,
        label_top=True,
        size_vertical=size_vertical,
    )
    ax.add_artist(bar)
    return ax