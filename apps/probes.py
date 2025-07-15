import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    C10 = mo.ui.slider(start=-1000, stop=1000, step=100, label="defocus [Å]", show_value=True)
    return C10, mo


@app.cell(hide_code=True)
def _(C10, fig, mo):
    mo.md(
        rf"""
    {mo.as_html(fig).center()}
    {C10.center()}
     """
    ).center()
    return


@app.cell
def _(C10):
    from probe import complex_probe, ssb_ctf
    from visualization import show_2d
    import torch
    import matplotlib.pyplot as plt
    from torch.fft import fft2, ifft2, fftshift, ifftshift
    from visualization_utils import ScalebarConfig


    n = 192
    q_max = 2 # inverse Angstroms
    sampling = 1 / q_max / 2 # Angstroms
    dk = q_max / (n/2)
    wavelength = 2.5e-2 # 200kV
    q_probe = 1
    qx = qy = torch.fft.fftfreq(n,sampling)
    q2 = qx[:,None]**2 + qy[None,:]**2
    q  = torch.sqrt(q2)
    C30 = 0


    images = []


    reciprocal_sampling = 2 * q_max / n
    probe_fourier = complex_probe(q, wavelength, C10.value, C30, q_probe, reciprocal_sampling)
    probe_reals = fftshift(ifft2(probe_fourier, norm="ortho"))

    scale1 = ScalebarConfig(
      sampling = dk,
      units = "A^-1",
      length = None,
      width_px = n/40,
      pad_px = 0.5,
      color = "white",
      loc = "lower right"
    )
    scale2 = ScalebarConfig(
      sampling = sampling/10,
      units = "nm",
      length = None,
      width_px = n/40,
      pad_px = 0.5,
      color = "white",
      loc = "lower right"
    )
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle(f"Defocus: {-C10.value/10} nm")
    _fig_, _ax_ = show_2d(fftshift(probe_fourier), title="Probe in Fourier Space", scalebar=scale1, figax=(fig,ax[0]))
    _fig_, _ax_ = show_2d(probe_reals, title= "Probe in Real Space", scalebar=scale2, figax=(fig,ax[1]))
    plt.tight_layout()
    plt.show()

      # Convert the whole figure to an RGB array
      # fig.canvas.draw()
      # rgba = np.asarray(fig.canvas.buffer_rgba()) 



    return (fig,)


if __name__ == "__main__":
    app.run()
