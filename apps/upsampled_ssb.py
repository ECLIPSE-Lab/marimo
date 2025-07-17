# %% [markdown]
# ---
# title: Upsampled SSB / Parallax Hybrid
# authors: [Julie Marie Bekkevold, Georgios Varnavides]
# date: 2025-04-09
# ---

# %%
# enable interactive matplotlib
%matplotlib widget 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import ctf # import custom plotting / utils
import cmasher as cmr
from tqdm.notebook import tqdm

import ipywidgets
from IPython.display import display

# %% [markdown]
# ## 4D STEM Simulation

# %%
# parameters
gpts = 192
k_max = 2 # inverse Angstroms
k_probe = 1 # inverse Angstroms
wavelength = 0.019687 # 300kV
sampling = 1 / k_max / 2 # Angstroms
reciprocal_sampling = 2 * k_max / gpts # inverse Angstroms

# slider_properties
C10 = 192
kde_upsample_factor = scan_step_size = 4
phi0 = 1.0

cmap = cmr.eclipse
sample_cmap = 'gray'

pixelated_ssb_line_color = 'darkgreen'
segmented_ssb_line_color = 'yellowgreen'
segmented_parallax_line_color = 'darksalmon'
pixelated_parallax_line_color = 'darkred'

# %% [markdown]
# ### White Noise Potential

# %%
def white_noise_object_2D(n, phi0):
    """ creates a 2D real-valued array, whose FFT has random phase and constant amplitude """

    evenQ = n%2 == 0
    
    # indices
    pos_ind = np.arange(1,(n if evenQ else n+1)//2)
    neg_ind = np.flip(np.arange(n//2+1,n))

    # random phase
    arr = np.random.randn(n,n)
    
    # top-left // bottom-right
    arr[pos_ind[:,None],pos_ind[None,:]] = -arr[neg_ind[:,None],neg_ind[None,:]]
    # bottom-left // top-right
    arr[pos_ind[:,None],neg_ind[None,:]] = -arr[neg_ind[:,None],pos_ind[None,:]]
    # kx=0
    arr[0,pos_ind] = -arr[0,neg_ind]
    # ky=0
    arr[pos_ind,0] = -arr[neg_ind,0]

    # zero-out components which don't have k-> -k mapping
    if evenQ:
        arr[n//2,:] = 0 # zero highest spatial freq
        arr[:,n//2] = 0 # zero highest spatial freq

    arr[0,0] = 0 # DC component

    # fourier-array
    arr = np.exp(2j*np.pi*arr)*phi0

    # inverse FFT and remove floating point errors
    arr = np.fft.ifft2(arr).real
    
    return arr

# potential
potential = white_noise_object_2D(gpts,phi0)
complex_obj = np.exp(1j*potential)

# %% [markdown]
# #### Import sample potentials

# %%
sto_potential = np.load("data/STO_projected-potential_192x192_4qprobe.npy")
sto_potential -= sto_potential.mean()
mof_potential = np.load("data/MOF_projected-potential_192x192_4qprobe.npy")
mof_potential -= mof_potential.mean()
apo_potential = np.load("data/apoF_projected-potential_192x192_4qprobe.npy")
apo_potential -= apo_potential.mean()
potentials = [sto_potential,mof_potential,apo_potential]

sto_sampling = 23.67 / sto_potential.shape[0]  # Å
mof_sampling = 4.48 / mof_potential.shape[0]  # nm
apo_sampling = 19.2 / apo_potential.shape[0]  # nm
samplings = [sto_sampling,mof_sampling,apo_sampling]
sampling_units = [r"$\AA$","nm","nm"]

# %% [markdown]
# ### Probe

# %%
def soft_aperture(k,k_probe=k_probe,reciprocal_sampling=reciprocal_sampling):
    """ """
    return np.sqrt(
        np.clip(
            (k_probe - k)/reciprocal_sampling + 0.5,
            0,
            1,
        ),
    )

kx = ky = np.fft.fftfreq(gpts,sampling)
k2 = kx[:,None]**2 + ky[None,:]**2
k = np.sqrt(k2)

def return_complex_probe(
    k,
    C10,
    wavelength=wavelength,
):
    """ """
    probe_array_fourier_0 = soft_aperture(k)
    probe_array_fourier_0 /= np.sqrt(np.sum(np.abs(probe_array_fourier_0)**2))
    return probe_array_fourier_0 * np.exp(-1j*np.pi*wavelength*C10*k**2)
    
def return_simulation_inputs(
    scan_step_size,
    C10,
):
    """ """
    
    complex_probe = return_complex_probe(
        k,
        C10
    )
    
    scan_gpts = gpts // scan_step_size
    scan_sampling = sampling * scan_step_size

    qx = qy = np.fft.fftfreq(scan_gpts,scan_sampling)
    q2 = qx[:,None]**2 + qy[None,:]**2
    q  = np.sqrt(q2)
    
    x = y = np.arange(0.0,gpts,scan_step_size)
    xx, yy = np.meshgrid(x,y,indexing='ij')
    positions = np.stack((xx.ravel(),yy.ravel()),axis=-1)
    row, col = ctf.return_patch_indices(positions,(gpts,gpts),(gpts,gpts))

    return [scan_gpts, qx, qy, row, col, complex_probe]

simulation_inputs = return_simulation_inputs(scan_step_size,C10)

# %%
def simulate_intensities(batch_size=None, pbar=None):
    """ """
    scan_gpts, qx, qy, row, col, complex_probe = simulation_inputs
    probe_array = np.fft.ifft2(complex_probe) * gpts
    
    if batch_size is None:
        batch_size = scan_gpts**2

    m = scan_gpts**2
    n_batch = int(m // batch_size)
    order = np.arange(m).reshape((n_batch,batch_size))
    intensities = np.zeros((m,gpts,gpts))

    if pbar is not None:
        pbar.reset(n_batch)
        pbar.colour = None
        pbar.refresh()

    for batch_index in range(n_batch):
        batch_order = order[batch_index]
        intensities[batch_order] = ctf.simulate_data(
            complex_obj,
            probe_array,
            row[batch_order],
            col[batch_order],
        )
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.colour = 'green'
        
    return intensities.reshape((scan_gpts,scan_gpts,gpts,gpts))**2 / gpts**2

intensities = [
    simulate_intensities(
        batch_size=256*6//scan_step_size,
        pbar=None,
    )
]

# %%
bf_mask = soft_aperture(k) > 0.5
kx_bf_indices,ky_bf_indices = np.where(bf_mask)
num_bf_pixels = bf_mask.sum()

kx_shift = -2j * np.pi * kx[:,None]
ky_shift = -2j * np.pi * ky[None,:]

def return_aperture_overlap_factor(
    unrolled_chi,
    k_qm,
    k_qp
):
    """ """
    aperture_minus = soft_aperture(k_qm)
    aperture_plus = soft_aperture(k_qp)
    
    beta = np.exp(-1j*unrolled_chi)*aperture_minus - np.exp(1j*unrolled_chi)*aperture_plus
    beta_abs = np.abs(beta)
    beta_abs[beta_abs==0] = np.inf
    aperture_overlap_factor = beta.conj() / beta_abs
    return aperture_overlap_factor

def return_full_gamma_factor(
    complex_probe_factor,
    k_qm,
    k_qp,
    C10,
):
    """ """
    shifted_probe_minus = return_complex_probe(k_qm,C10)
    shifted_probe_plus = return_complex_probe(k_qp,C10)
    gamma = shifted_probe_minus*complex_probe_factor.conjugate() - shifted_probe_plus.conjugate()*complex_probe_factor
    return gamma

def pad_scan_sampled_ctf(
    ctf_array,
):
    """ """
    pad_width = (gpts - ctf_array.shape[0])//2
    return np.fft.ifftshift(np.pad(np.fft.fftshift(ctf_array),pad_width))

def return_reconstruction_ctfs(
    scan_step_size,
    C10,
    pbar=None
):
    """ """

    vbfs = intensities[0][...,bf_mask]
    vbfs = vbfs / vbfs.mean((0,1)) - 1
    vbfs = np.moveaxis(vbfs,(0,1,2),(1,2,0))  

    scan_gpts, qx, qy, row, col, complex_probe = simulation_inputs
    
    q2 = qx[:,None]**2 + qy[None,:]**2
    unrolled_chi = np.pi*wavelength*C10*q2
    sign_sin_chi = np.sign(np.sin(unrolled_chi))
    
    unrolled_chi_k = np.pi*wavelength*C10*k2
    sign_sin_chi_k = np.sign(np.sin(unrolled_chi_k))

    qx_shift = -2j * np.pi * qx[:,None]
    qy_shift = -2j * np.pi * qy[None,:]
    
    grad_x, grad_y = np.meshgrid(
        kx * wavelength * C10,
        ky * wavelength * C10,
        indexing='ij'
    )
    
    shifts_ang = np.stack((grad_x[bf_mask],grad_y[bf_mask]),-1)

    parallax_recon = np.zeros((scan_gpts,scan_gpts))
    upsampled_parallax_recon = np.zeros((gpts,gpts))
    
    ssb_recon = np.zeros((scan_gpts,scan_gpts))
    upsampled_ssb_recon = np.zeros((gpts,gpts))

    if pbar is not None:
        pbar.reset(num_bf_pixels)
        pbar.colour = None
        pbar.refresh()
    
    for bf_index in range(num_bf_pixels):

        # scan-sampled parallax
        G = np.fft.fft2(vbfs[bf_index])
        shift_op = np.exp(
            qx_shift * shifts_ang[bf_index,0]
            + qy_shift * shifts_ang[bf_index,1]
        )
        
        parallax_recon += np.fft.ifft2(G*shift_op*sign_sin_chi).real

        # upsampled parallax
        tiled_G = np.tile(G,(scan_step_size,scan_step_size))
        upsampled_shift_op = np.exp(
            kx_shift * shifts_ang[bf_index,0]
            + ky_shift * shifts_ang[bf_index,1]
        )

        upsampled_parallax_recon += np.fft.ifft2(tiled_G*upsampled_shift_op*sign_sin_chi_k).real

        # scan-sampled ssb
        kx_ind = kx_bf_indices[bf_index]
        ky_ind = ky_bf_indices[bf_index]

        kx_qx = qx - kx[kx_ind]
        ky_qy = qy - ky[ky_ind]
        k_qm = np.sqrt(kx_qx[:,None]**2 + ky_qy[None,:]**2)
        
        kx_qx = qx + kx[kx_ind]
        ky_qy = qy + ky[ky_ind]
        k_qp = np.sqrt(kx_qx[:,None]**2 + ky_qy[None,:]**2)
        
        aperture_overlap_factor = return_aperture_overlap_factor(
            unrolled_chi,
            k_qm,
            k_qp
        )
        ssb_recon += np.fft.ifft2(G*shift_op*aperture_overlap_factor).imag

        # upsampled ssb
        kx_qx = kx - kx[kx_ind]
        ky_qy = ky - ky[ky_ind]
        k_qm = np.sqrt(kx_qx[:,None]**2 + ky_qy[None,:]**2)
        
        kx_qx = kx + kx[kx_ind]
        ky_qy = ky + ky[ky_ind]
        k_qp = np.sqrt(kx_qx[:,None]**2 + ky_qy[None,:]**2)
        
        aperture_overlap_factor = return_aperture_overlap_factor(
            unrolled_chi_k,
            k_qm,
            k_qp
        )
        upsampled_ssb_recon += np.fft.ifft2(tiled_G*upsampled_shift_op*aperture_overlap_factor).imag

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.colour = 'green'

    normalization_factor = num_bf_pixels / scan_step_size**2
    parallax_recon /= normalization_factor
    upsampled_parallax_recon /= normalization_factor
    ssb_recon /= normalization_factor
    upsampled_ssb_recon /= normalization_factor

    ctf_arrays = [
        pad_scan_sampled_ctf(ctf.compute_ctf(parallax_recon)),
        ctf.compute_ctf(upsampled_parallax_recon),
        pad_scan_sampled_ctf(ctf.compute_ctf(ssb_recon)),
        ctf.compute_ctf(upsampled_ssb_recon)
    ]
    
    return ctf_arrays

# %%
# ctf_arrays = return_reconstruction_ctfs(
#     scan_step_size,
#     C10,
#     None
# )

ctf_arrays = [np.zeros((gpts,gpts))]*4

# %%
def return_binned_padded_ctfs(ctf_array):
    binned_ctf = ctf_array.reshape((gpts//2,2,gpts//2,2)).sum((1,3))
    zero_pad_ctf_to_4qprobe = np.fft.ifftshift(np.pad(np.fft.fftshift(binned_ctf),gpts//4))
    return zero_pad_ctf_to_4qprobe

def return_aperture_factors(
    C10,
    kx_index = 10,
    ky_index = 20,
):
    """ """
    unrolled_chi_k = np.pi*wavelength*C10*k2
    complex_probe_factor = simulation_inputs[-1][kx_index,ky_index]
    
    kx_qx = kx - kx[kx_index]
    ky_qy = ky - ky[ky_index]
    k_qm = np.sqrt(kx_qx[:,None]**2 + ky_qy[None,:]**2)
    
    kx_qx = kx + kx[kx_index]
    ky_qy = ky + ky[ky_index]
    k_qp = np.sqrt(kx_qx[:,None]**2 + ky_qy[None,:]**2)
    
    aperture_overlap_factor = return_aperture_overlap_factor(
        unrolled_chi_k,
        k_qm,
        k_qp
    )
    
    full_gamma_factor = return_full_gamma_factor(
        complex_probe_factor,
        k_qm,
        k_qp,
        C10
    )

    return full_gamma_factor, aperture_overlap_factor

# %%
with plt.ioff():
    dpi=72
    fig, axs = plt.subplots(2,4,figsize=(640/dpi,365/dpi),dpi=dpi)

par_split_ctf = np.fft.fftshift(ctf_arrays[0])
par_split_ctf[:,gpts//2:] = np.fft.fftshift(ctf_arrays[1])[:,gpts//2:]

ssb_split_ctf = np.fft.fftshift(ctf_arrays[2])
ssb_split_ctf[:,gpts//2:] = np.fft.fftshift(ctf_arrays[3])[:,gpts//2:]

im_ctfs = []
for ax, arr, title in zip(axs[0,:2],[par_split_ctf,ssb_split_ctf],["tcBF CTF","SSB CTF"]):
    im = ax.imshow(arr/2,vmin=0,vmax=1,cmap=cmap)
    im.set_alpha(0.25)
    im_ctfs.append(im)
    ax.vlines(gpts//2,0,gpts-1,colors='w')
    ax.set(xticks=[],yticks=[],title=title)
    ax.text(4,4,"scan-\nsampled",color="white",va='top',ha='left')
    ax.text(gpts-4,4,"up-\nsampled",color="white",va='top',ha='right')
    ctf.add_scalebar(ax,length=gpts//4,sampling=reciprocal_sampling,units=r'$q_{\mathrm{probe}}$')

# apo_imgs = [
#     ctf.histogram_scaling(
#         np.fft.ifft2(
#             np.fft.fft2(apo_potential) * return_binned_padded_ctfs(ctf_array)).real,
#         normalize=True
#     ) for ctf_array in ctf_arrays
# ]

# par_apo = apo_imgs[0].copy()
# par_apo[:,gpts//2:] = apo_imgs[1][:,gpts//2:].copy()

# ssb_apo = apo_imgs[2].copy()
# ssb_apo[:,gpts//2:] = apo_imgs[3][:,gpts//2:].copy()

par_apo = np.zeros_like(apo_potential)
par_apo[0,0] = 1
ssb_apo = par_apo.copy()

im_wpos = [] 
for ax, arr, title in zip(axs[1,:2],[par_apo,ssb_apo],["tcBF-convolved WPO","SSB-convolved WPO"]):
    im = ax.imshow(arr,cmap=sample_cmap)
    im.set_alpha(0.25)
    im_wpos.append(im)
    ax.vlines(gpts//2,0,gpts-1,colors='w')
    ax.set(xticks=[],yticks=[],title=title)
    ax.text(4,4,"scan-\nsampled",color="white",va='top',ha='left')
    ax.text(gpts-4,4,"up-\nsampled",color="white",va='top',ha='right')
    ctf.add_scalebar(ax,length=gpts//5,sampling=sto_sampling,units=r'$\AA$')
    ctf.add_scalebar(ax,length=gpts//5,sampling=mof_sampling,units=r'nm')
    ctf.add_scalebar(ax,length=gpts//5,sampling=apo_sampling,units=r'nm')

sto_scalebar_0, mof_scalebar_0, apo_scalebar_0 = axs[1,0].artists
sto_scalebar_1, mof_scalebar_1, apo_scalebar_1 = axs[1,1].artists
sto_scalebar_0.set_visible(False)
sto_scalebar_1.set_visible(False)
mof_scalebar_0.set_visible(False)
mof_scalebar_1.set_visible(False)

aperture_factors = return_aperture_factors(C10)
im_factors = []
for ax, arr, title in zip(axs[0,2:],aperture_factors,["aperture overlap","shift decoupled"]):
    im = ax.imshow(ctf.complex_to_rgb(np.fft.fftshift(arr)))
    im_factors.append(im)
    ax.set(xticks=[],yticks=[],title=title)
    ctf.add_scalebar(ax,length=gpts//4,sampling=reciprocal_sampling,units=r'$q_{\mathrm{probe}}$')

gs = axs[1, 2].get_gridspec()
for ax in axs[1,2:]:
    ax.remove()
axbig = fig.add_subplot(gs[1,2:])

qI_bins = [
    ctf.radially_average_ctf(ctf_array/2,(sampling,sampling))
    for ctf_array in ctf_arrays
]

plot_tcbf_u = axbig.plot(*qI_bins[1],label="up-sampled tcBF",color=pixelated_parallax_line_color)[0]
plot_tcbf_s = axbig.plot(
    qI_bins[0][0][:gpts//scan_step_size//2],
    qI_bins[0][1][:gpts//scan_step_size//2],
    label="scan-sampled tcBF",
    color=segmented_parallax_line_color
)[0]
plot_ssb_u = axbig.plot(*qI_bins[3],label="up-sampled SSB",color=pixelated_ssb_line_color)[0]
plot_ssb_s = axbig.plot(
    qI_bins[2][0][:gpts//scan_step_size//2],
    qI_bins[2][1][:gpts//scan_step_size//2],
    label="scan-sampled SSB",
    color=segmented_ssb_line_color
)[0]
vline = axbig.vlines(k_max/scan_step_size,0,1,colors='k',linestyles='--',linewidth=1,)

axbig.set(
    yticks=[],
    xlabel=r"spatial frequency, $q/q_{\mathrm{probe}}$",
    ylim=[0,1],
    xlim=[0,k_max],
    title="radially averaged tcBF and SSB CTFs"
)
axbig.legend()

for plot in [plot_tcbf_s,plot_tcbf_u,plot_ssb_s,plot_ssb_u]:
    plot.set_alpha(0.25)

fig.tight_layout()
fig.canvas.resizable = False
fig.canvas.header_visible = False
fig.canvas.footer_visible = False
fig.canvas.toolbar_visible = False
# fig.canvas.toolbar_visible = True
# fig.canvas.toolbar_position = 'bottom'
fig.canvas.layout.width = '640px'
fig.canvas.layout.height = '380px'
fig.tight_layout()
None

# %%
style = {'description_width': 'initial'}
layout_quarter = ipywidgets.Layout(width="160px",height="30px")
layout_half = ipywidgets.Layout(width="320px",height="30px")

kwargs_quarter = {'style':style,'layout':layout_quarter,'continuous_update':False}
kwargs_half = {'style':style,'layout':layout_half,'continuous_update':False}

defocus_slider = ipywidgets.IntSlider(
    value = gpts, min = -gpts, max = gpts, step = 1,
    description = "negative defocus, $C_{1,0}$ [Å]",
    **kwargs_half
)

subsampling_slider = ipywidgets.Dropdown(
    options=[2,3,4,6],
    value=4,
    description="sub-sampling",
    **kwargs_quarter
)

object_dropdown = ipywidgets.Dropdown(
    options=[("strontium titanate",0),("metal-organic framework",1),("apoferritin protein",2)],
    value=2,
    **kwargs_quarter
)

simulate_button = ipywidgets.Button(
    description='simulate (expensive)',
    **kwargs_quarter
)

simulation_pbar = tqdm(total=12,display=False)
simulation_pbar_wrapper = ipywidgets.HBox(simulation_pbar.container.children[:2],**kwargs_quarter)

reconstruct_button = ipywidgets.Button(
    description='reconstruct (expensive)',
    **kwargs_quarter
)
reconstruct_button.button_style = 'warning'

reconstruction_pbar = tqdm(total=num_bf_pixels,display=False)
reconstruction_pbar_wrapper = ipywidgets.HBox(reconstruction_pbar.container.children[:2],**kwargs_quarter)

def disable_all(boolean):
    defocus_slider.disabled = boolean
    object_dropdown.disabled = boolean
    simulate_button.disabled = boolean
    reconstruct_button.disabled = boolean
    return None

def defocus_wrapper(*args):
    for im in im_ctfs:
        im.set_alpha(0.25)
    for im in im_wpos:
        im.set_alpha(0.25)
    for plot in [plot_tcbf_s,plot_tcbf_u,plot_ssb_s,plot_ssb_u]:
        plot.set_alpha(0.25)

    aperture_factors = return_aperture_factors(defocus_slider.value)
    for im, arr in zip(im_factors,aperture_factors):
        im.set_data(ctf.complex_to_rgb(np.fft.fftshift(arr)))
        
    simulate_button.button_style = 'warning'
    simulation_pbar.reset()
    
defocus_slider.observe(defocus_wrapper,names='value')
subsampling_slider.observe(defocus_wrapper,names='value')

def simulate_wrapper(*args):
    disable_all(True)
    simulate_and_update_panels(
        subsampling_slider.value,
        defocus_slider.value,
        simulation_pbar,
    )
    disable_all(False)
    reconstruct_button.button_style = 'warning'
    reconstruction_pbar.reset()
    simulate_button.button_style = ''
    disable_all(False)
    
simulate_button.on_click(simulate_wrapper)

def simulate_and_update_panels(
    scan_step_size,
    C10,
    pbar,
):
    """ """
    simulation_inputs[:] = return_simulation_inputs(
        scan_step_size,
        C10,
    )
    intensities[0] = simulate_intensities(
        batch_size=256*6//scan_step_size,
        pbar=pbar,
    )
    
def reconstruct_wrapper(*args):
    disable_all(True)
    reconstruct_and_update_panels(
        subsampling_slider.value,
        defocus_slider.value,
        pbar=reconstruction_pbar
    )
    disable_all(False)

reconstruct_button.on_click(reconstruct_wrapper)

def update_wpo_panels(*args):
    obj = potentials[object_dropdown.value]    
    obj_imgs = [
        ctf.histogram_scaling(
            np.fft.ifft2(
                np.fft.fft2(obj) * return_binned_padded_ctfs(ctf_array)).real,
            normalize=True
        ) for ctf_array in ctf_arrays
    ]
    
    par_obj = obj_imgs[0].copy()
    par_obj[:,gpts//2:] = obj_imgs[1][:,gpts//2:].copy()
    
    ssb_obj = obj_imgs[2].copy()
    ssb_obj[:,gpts//2:] = obj_imgs[3][:,gpts//2:].copy()

    for im, arr in zip(im_wpos,[par_obj,ssb_obj]):
        im.set_data(arr)
        im.set_alpha(1)

    sto_scalebar_0.set_visible(object_dropdown.value == 0)
    sto_scalebar_1.set_visible(object_dropdown.value == 0)
    mof_scalebar_0.set_visible(object_dropdown.value == 1)
    mof_scalebar_1.set_visible(object_dropdown.value == 1)
    apo_scalebar_0.set_visible(object_dropdown.value == 2)
    apo_scalebar_1.set_visible(object_dropdown.value == 2)

object_dropdown.observe(update_wpo_panels,names='value')
        
def reconstruct_and_update_panels(scan_step_size,C10,pbar):
    """ """
    ctf_arrays[:] = return_reconstruction_ctfs(
        scan_step_size,
        C10,
        pbar,
    )

    par_split_ctf = np.fft.fftshift(ctf_arrays[0])
    par_split_ctf[:,gpts//2:] = np.fft.fftshift(ctf_arrays[1])[:,gpts//2:]
    
    ssb_split_ctf = np.fft.fftshift(ctf_arrays[2])
    ssb_split_ctf[:,gpts//2:] = np.fft.fftshift(ctf_arrays[3])[:,gpts//2:]

    for im, arr in zip(im_ctfs,[par_split_ctf,ssb_split_ctf]):
        im.set_data(arr/2)
        im.set_alpha(1)
            
    qI_bins = [
        ctf.radially_average_ctf(ctf_array/2,(sampling,sampling))
        for ctf_array in ctf_arrays
    ]

    plot_tcbf_s.set_xdata(qI_bins[0][0][:gpts//scan_step_size//2])
    plot_tcbf_u.set_xdata(qI_bins[1][0])
    plot_ssb_s.set_xdata(qI_bins[2][0][:gpts//scan_step_size//2])
    plot_ssb_u.set_xdata(qI_bins[3][0])
    
    plot_tcbf_s.set_ydata(qI_bins[0][1][:gpts//scan_step_size//2])
    plot_tcbf_u.set_ydata(qI_bins[1][1])
    plot_ssb_s.set_ydata(qI_bins[2][1][:gpts//scan_step_size//2])
    plot_ssb_u.set_ydata(qI_bins[3][1])

    vline.set_paths([np.array([[k_max/scan_step_size,0],[k_max/scan_step_size,1]])])

    for plot in [plot_tcbf_s,plot_tcbf_u,plot_ssb_s,plot_ssb_u]:
        plot.set_alpha(1)

    update_wpo_panels()
        
    reconstruct_button.button_style = ''

    fig.canvas.draw_idle()
    return None

# %%
#| label: app:upsampled_ssb
display(
    ipywidgets.VBox(
        [
            ipywidgets.VBox(
                [
                    ipywidgets.HBox([defocus_slider,simulate_button,simulation_pbar_wrapper]),
                    ipywidgets.HBox([subsampling_slider,object_dropdown,reconstruct_button,reconstruction_pbar_wrapper]),
                ]
            ),
            fig.canvas
        ]
    )
)

# %%



