#%%
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

#%%