import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    # controls
    max_semiangle = mo.ui.dropdown(options={r"1 alpha":1, "2 alpha":2, "3 alpha":3, "4 alpha":4, "5 alpha":5, "6 alpha":6},
                            value="2 alpha", # initial value
                            label="Max. Semiangle")
    energy = mo.ui.dropdown(options={"300 keV":300e3, "200 keV":200e3, "80 keV":80e3, "60 keV":60e3, "30 keV":30e3},
                            value="300 keV", # initial value
                            label="Electron energy")
    detector = mo.ui.dropdown(options={"512":512,"256":256, "192":192, "128":128, "96":96, "48":48, "16":16},
                            value="512", # initial value
                            label="Detector shape")



    astigmatism = mo.ui.slider(
        start=0, stop=200, step=5, label="astigmatism [Å]", show_value=True
    )

    astigmatism_angle_slider = mo.ui.slider(
        start=-90, stop=90, step=1, label="astigmatism angle [°]", show_value=True
    )

    convergence_angle = mo.ui.slider(
        start=5,
        stop=40,
        value=20,
        step=0.5,
        label="conv. semiangle [mrad]",
        show_value=True,
    )


    # mo.md(
    #     f"""

    #     {convergence_angle.center()}
    #     {defocus.center()}
    #     {astigmatism.center()}
    #     {astigmatism_angle.center()}
    #     {mo.as_html(fig).center()}
    #     {mo.as_html(fig_metrics).center()}

    # """
    # ).center()
    return (
        astigmatism,
        astigmatism_angle_slider,
        convergence_angle,
        detector,
        energy,
        max_semiangle,
    )


@app.cell
def _(
    convergence_angle,
    detector,
    energy,
    energy2wavelength,
    max_semiangle,
    mo,
    np,
):
    lam = energy2wavelength(energy.value)
    max_detector_alpha = convergence_angle.value * max_semiangle.value * 1e-3
    k_max = max_detector_alpha / lam 
    sampling = 1 / 2 / k_max 
    dk = 1 / (detector.value * sampling)
    FOV = 1 / dk 
    max_defocus = (FOV/2 ) / np.tan(convergence_angle.value * 1e-3)
    # defocus.start = - max_defocus
    # defocus.stop = max_defocus

    defocus = mo.ui.slider(
        start=0, stop=max_defocus, step=5, label="defocus [Å]", show_value=True, value = 100
    )
    scan_step = sampling
    return defocus, dk, lam, max_detector_alpha, sampling, scan_step


@app.cell
def _(
    ComplexProbe,
    astigmatism,
    astigmatism_angle_slider,
    convergence_angle,
    defocus,
    detector,
    energy,
    np,
    sampling,
):
    probe = ComplexProbe(
        energy1=energy.value,
        gpts=(detector.value, detector.value),
        sampling=(sampling, sampling),
        semiangle_cutoff=convergence_angle.value,
        defocus=defocus.value,
        astigmatism=astigmatism.value,
        astigmatism_angle=np.deg2rad(astigmatism_angle_slider.value),

    ).build() 
    probe_real = np.fft.fftshift(probe._array)
    probe_fourier = np.fft.fftshift(probe._array_fourier)
    return probe, probe_fourier, probe_real


@app.cell
def _(
    add_scalebar,
    convergence_angle,
    defocus,
    detector,
    dk,
    energy,
    fig_g,
    fig_pctf,
    lam,
    max_detector_alpha,
    max_semiangle,
    mo,
    np,
    plt,
    probe,
    probe_fourier,
    probe_real,
    sampling,
    scan_step,
    show_complex,
):

    probe_radius = np.abs(defocus.value) * np.tan(convergence_angle.value * 1e-3)
    areal_oversampling = np.pi * probe_radius**2/scan_step**2
    dalpha = dk * lam  

    ronchi_mag = convergence_angle.value * 1e-3 / ( dalpha  * probe_radius )
    fund_samp = 1 / (dk * scan_step)
    probe_sampling = 1 / (2 * probe_radius * dk)

    metrics = {
        'wavelength': lam,
        'probe_radius': probe_radius,
        'scan_step': scan_step,
        'diff_step': dk,
        'ronchi_mag': ronchi_mag,
        'areal_oversamp': areal_oversampling,
        'fund_samp': fund_samp
    }
 

    print(f'probe_radius      : {probe_radius}')
    print(f'probe_sampling    : {probe_sampling}')
    print(f'ronchi_mag        : {ronchi_mag}')
    print(f'areal_oversamp    : {areal_oversampling}')
    print(f'fund_samp         : {fund_samp}')
    print()
    print(f'lam               : {lam}') 
    print(f'det shape         : {detector.value}')
    print(f'max_detector_alpha: {max_detector_alpha*1e3}')
    print(f'dk                : {dk}')
    print(f'dk probe          : {probe.reciprocal_space_sampling}')
    print(f'sampling          : {sampling}')

    print(f'max_detector_alpha: {max_detector_alpha}')
    print(f'dalpha                    [mrad]: {dalpha*1e3}')
    print(f'probe.angular_sampling[0] [mrad]: {probe.angular_sampling}')


    fig, (ax_fourier, ax_real) = plt.subplots(1, 2, figsize=(8, 4))
    ax_real = show_complex(
        probe_real,
        figax=(fig, ax_real),
        ticks=False,
        vmin=0,
        vmax=1,
    )
    ax_fourier = show_complex(
        probe_fourier,
        figax=(fig, ax_fourier),
        ticks=False,
        vmin=0,
        vmax=1,
    )
    ax_real.set_title("real-space wave")
    ax_fourier.set_title("reciprocal-space wave")
    add_scalebar(ax_real, probe.sampling[0], r"$\AA$")
    add_scalebar(ax_fourier, probe.angular_sampling[0], "mrad")
    text_nyquist = mo.md(
        f"""
        Nyquist res.      : {2*sampling:2.2f}Å 
    """
    )
    text1 = mo.md(
        f"""
        Probe Radius      : {probe_radius:2.2f}Å 
    """
    )
    text2 = mo.md(
        f""" 
        Probe Sampling    : {probe_sampling:2.2f}

    """
    ) 
    text3 = mo.md(
        f""" 
        Ronchigram Magn.  : {ronchi_mag:2.2f} 
    """
    ) 
    text4 = mo.md(
        f""" 
        Areal oversampling: {areal_oversampling:2.2f} 
    """
    ) 
    text5 = mo.md(
        f""" 
        References: 

        - Yang et al. (2016) [10/f9fwhk](https://doi.org/10/f9fwhk)
 

    """
    ).center()
    text6 = mo.md(
        f"""    
 
    """
    ).center()
    text7 = mo.md(
        f""" 
        scan_step: {scan_step:2.2f} 
    """
    ) 
 
    vertical = mo.vstack(
        [energy,max_semiangle,detector,convergence_angle,defocus, text_nyquist,text7,text1, text2, text3, text4],
        align="end",
        gap=0
    )
    horizontal = mo.hstack(
        [ vertical, fig],
        align="center",
        justify="center",
        gap=0,
        wrap=False,
    )
    horizontal2 = mo.hstack(
        [ text5],
        align="center",
        justify="center",
        gap=0,
        wrap=False,
    )
    vertical2 = mo.vstack(
        [horizontal, fig_pctf,fig_g, horizontal2],
        align="start",
        gap=0
    )

    vertical2
    return


@app.cell
def _(convergence_angle, np, plt, probe, probe_fourier, show_complex):
    sh = np.array(probe_fourier.shape)

    Asum = np.abs(probe_fourier).sum()
    pctf = np.zeros((sh[0]//2,))

    plot_gammas = []
    nrange = sh[0]//2
    facs = [8,4,2,4/3]
    for nroll in range(nrange):
        psi_plus = np.roll(probe_fourier, (0,-nroll))
        psi_minus = np.roll(probe_fourier, (0,nroll))
        gamma = probe_fourier.conj() * psi_minus - probe_fourier * psi_plus.conj()
        gammasum = np.abs(gamma).sum()
        pctf[nroll] = 0.5 * gammasum / Asum
        if nroll in [nrange//facs[i] for i in range(4)]:
            plot_gammas.append(gamma)

    fig_g, ax_gammas = plt.subplots(1, 4, figsize=(16/1.5, 3/1.5))
    for i, gamma in enumerate(plot_gammas):
        ax_gammas[i] = show_complex(
            gamma[sh[0]//8:-sh[0]//8,sh[1]//8:-sh[1]//8],
            figax=(fig_g, ax_gammas[i]),
            ticks=False,
            vmin=0,
            vmax=1,
        )    
        ax_gammas[i].set_title(f'G at {convergence_angle.value/facs[i]*2:2.2f}mrad')
        # ax_gammas[1,i].imshow(np.abs(gamma[sh[0]//8:-sh[0]//8,sh[1]//8:-sh[1]//8]), cmap='bone')
        # ax_gammas[1,i].set_xticks([])
        # ax_gammas[1,i].set_yticks([])
        # ax_gammas[1,i].set_title(f'|G| at {convergence_angle.value/facs[i]*2:2.2f}mrad')
    fig_g.tight_layout() 
    plt.show()


    kx, ky = probe.get_spatial_frequencies()
    alphax, alphay = kx * probe._wavelength, ky * probe._wavelength
    alpha = np.sqrt(alphax[:, None] ** 2 + alphay[None, :] ** 2)

    fig_pctf, ax_pctf = plt.subplots(figsize=(16.8/1.5,3))
    ax_pctf.plot(pctf)
    ax_pctf.set_ylabel('PCTF')
    ax_pctf.set_xlabel('spatial frequency [1/Å]')
    ax_pctf.set_ylim(0, 1) 
    ax_pctf.set_xlim(0, sh[0]/2) 

    # Set 16 evenly spaced x-ticks
    xtick_positions = np.linspace(0, sh[0]/2, 16)
    ax_pctf.set_xticks(xtick_positions)
    w =  kx[:sh[0]//2][::sh[0]//32]
 
    ax_pctf.set_xticklabels([f'{x:.1f}' for x in w])

    ax_top = ax_pctf.twiny()
    w =  kx[:sh[0]//2][::sh[0]//32]
    w = w[1:]
    xtick_positions = np.linspace(0, sh[0]/2, 16)
    ax_top.set_xticks(xtick_positions[1:])
    ax_top.set_xticklabels([f'{1/x:.1f}' for x in w])
    ax_top.set_xlabel('spatial distances [Å]')
    plt.show()
    return fig_g, fig_pctf, sh


@app.cell
def _(sh):
    sh[0]//16
    return


@app.cell
def _(np):
    # Complex Probes Utilities
    def energy2wavelength(energy1):
        """ """
        hplanck = 6.62607e-34
        c = 299792458.0
        me = 9.1093856e-31
        e = 1.6021766208e-19

        return (
            hplanck
            * c
            / np.sqrt(energy1 * (2 * me * c**2 / e + energy1))
            / e
            * 1.0e10
        )


    class ComplexProbe:
        """ """

        # fmt: off
        _polar_symbols = (
            "C10", "C12", "phi12",
            "C21", "phi21", "C23", "phi23",
            "C30", "C32", "phi32", "C34", "phi34",
            "C41", "phi41", "C43", "phi43", "C45", "phi45",
            "C50", "C52", "phi52", "C54", "phi54", "C56", "phi56",
        )

        _polar_aliases = {
            "defocus": "C10", "astigmatism": "C12", "astigmatism_angle": "phi12",
            "coma": "C21", "coma_angle": "phi21",
            "Cs": "C30",
            "C5": "C50",
        }
        # fmt: on

        def __init__(
            self,
            energy1,
            gpts,
            sampling,
            semiangle_cutoff,
            soft_aperture=True,
            parameters={},
            **kwargs,
        ):
            self._energy = energy1
            self._gpts = gpts
            self._sampling = sampling
            self._semiangle_cutoff = semiangle_cutoff
            self._soft_aperture = soft_aperture

            self._parameters = dict(
                zip(self._polar_symbols, [0.0] * len(self._polar_symbols))
            )
            parameters.update(kwargs)
            self.set_parameters(parameters)
            self._wavelength = energy2wavelength(self._energy)

        def set_parameters(self, parameters):
            """ """
            for symbol, value in parameters.items():
                if symbol in self._parameters.keys():
                    self._parameters[symbol] = value

                elif symbol == "defocus":
                    self._parameters[self._polar_aliases[symbol]] = -value

                elif symbol in self._polar_aliases.keys():
                    self._parameters[self._polar_aliases[symbol]] = value

                else:
                    raise ValueError(
                        "{} not a recognized parameter".format(symbol)
                    )

            return parameters

        def get_spatial_frequencies(self):
            return tuple(
                np.fft.fftfreq(n, d) for n, d in zip(self._gpts, self._sampling)
            )

        def get_scattering_angles(self):
            kx, ky = self.get_spatial_frequencies()
            kx, ky = kx * self._wavelength, ky * self._wavelength
            alpha = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
            phi = np.arctan2(ky[None, :], kx[:, None])
            return alpha, phi

        def hard_aperture(self, alpha, semiangle_cutoff):
            return alpha <= semiangle_cutoff

        def soft_aperture(self, alpha, semiangle_cutoff, angular_sampling):
            denominator = (
                np.sqrt(angular_sampling[0] ** 2 + angular_sampling[1] ** 2) * 1e-3
            )
            return np.clip((semiangle_cutoff - alpha) / denominator + 0.5, 0, 1)

        def evaluate_aperture(self, alpha, phi):
            if self._soft_aperture:
                return self.soft_aperture(
                    alpha, self._semiangle_cutoff * 1e-3, self.angular_sampling
                )
            else:
                return self.hard_aperture(alpha, self._semiangle_cutoff * 1e-3)

        def evaluate_chi(self, alpha, phi):
            p = self._parameters

            alpha2 = alpha**2

            array = np.zeros_like(alpha)
            if any([p[symbol] != 0.0 for symbol in ("C10", "C12", "phi12")]):
                array += (
                    1
                    / 2
                    * alpha2
                    * (p["C10"] + p["C12"] * np.cos(2 * (phi - p["phi12"])))
                )

            if any(
                [p[symbol] != 0.0 for symbol in ("C21", "phi21", "C23", "phi23")]
            ):
                array += (
                    1
                    / 3
                    * alpha2
                    * alpha
                    * (
                        p["C21"] * np.cos(phi - p["phi21"])
                        + p["C23"] * np.cos(3 * (phi - p["phi23"]))
                    )
                )

            if any(
                [
                    p[symbol] != 0.0
                    for symbol in ("C30", "C32", "phi32", "C34", "phi34")
                ]
            ):
                array += (
                    1
                    / 4
                    * alpha2**2
                    * (
                        p["C30"]
                        + p["C32"] * np.cos(2 * (phi - p["phi32"]))
                        + p["C34"] * np.cos(4 * (phi - p["phi34"]))
                    )
                )

            if any(
                [
                    p[symbol] != 0.0
                    for symbol in ("C41", "phi41", "C43", "phi43", "C45", "phi41")
                ]
            ):
                array += (
                    1
                    / 5
                    * alpha2**2
                    * alpha
                    * (
                        p["C41"] * np.cos((phi - p["phi41"]))
                        + p["C43"] * np.cos(3 * (phi - p["phi43"]))
                        + p["C45"] * np.cos(5 * (phi - p["phi45"]))
                    )
                )

            if any(
                [
                    p[symbol] != 0.0
                    for symbol in (
                        "C50",
                        "C52",
                        "phi52",
                        "C54",
                        "phi54",
                        "C56",
                        "phi56",
                    )
                ]
            ):
                array += (
                    1
                    / 6
                    * alpha2**3
                    * (
                        p["C50"]
                        + p["C52"] * np.cos(2 * (phi - p["phi52"]))
                        + p["C54"] * np.cos(4 * (phi - p["phi54"]))
                        + p["C56"] * np.cos(6 * (phi - p["phi56"]))
                    )
                )

            array = 2 * np.pi / self._wavelength * array
            return array

        def evaluate_aberrations(self, alpha, phi):
            return np.exp(-1.0j * self.evaluate_chi(alpha, phi))

        def evaluate_ctf(self):
            alpha, phi = self.get_scattering_angles()
            array = self.evaluate_aberrations(alpha, phi)
            array *= self.evaluate_aperture(alpha, phi)
            return array

        def build(self):
            self._array_fourier = self.evaluate_ctf()
            array = np.fft.ifft2(self._array_fourier)
            array /= np.sqrt(np.sum(np.abs(array) ** 2))
            self._array = array
            return self

        @property
        def sampling(self):
            return self._sampling

        @property
        def reciprocal_space_sampling(self):
            return tuple(1 / (n * s) for n, s in zip(self._gpts, self._sampling))

        @property
        def angular_sampling(self):
            return tuple(
                dk * self._wavelength * 1e3
                for dk in self.reciprocal_space_sampling
            )


    return ComplexProbe, energy2wavelength


@app.cell
def _(AnchoredSizeBar, cspace_convert, np, plt):
    # Complex Plotting Utilities
    def Complex2RGB(
        complex_data, vmin=None, vmax=None, power=None, chroma_boost=1
    ):
        """ """
        amp = np.abs(complex_data)
        phase = np.angle(complex_data)

        if power is not None:
            amp = amp**power

        if np.isclose(np.max(amp), np.min(amp)):
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = np.max(amp)
        else:
            if vmin is None:
                vmin = 0.02
            if vmax is None:
                vmax = 0.98
            vals = np.sort(amp[~np.isnan(amp)])
            ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
            ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
            ind_vmin = np.max([0, ind_vmin])
            ind_vmax = np.min([len(vals) - 1, ind_vmax])
            vmin = vals[ind_vmin]
            vmax = vals[ind_vmax]

        amp = np.where(amp < vmin, vmin, amp)
        amp = np.where(amp > vmax, vmax, amp)
        amp = ((amp - vmin) / vmax).clip(1e-16, 1)

        J = amp * 61.5  # Note we restrict luminance to the monotonic chroma cutoff
        C = np.minimum(chroma_boost * 98 * J / 123, 110)
        h = np.rad2deg(phase) + 180

        JCh = np.stack((J, C, h), axis=-1)
        rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

        return rgb


    def add_scalebar(ax, sampling, units):
        """ """
        bar = AnchoredSizeBar(
            ax.transData,
            20,
            f"{np.round(sampling,1)*20:.0f} {units}",
            "lower right",
            pad=0.5,
            color="white",
            frameon=False,
            label_top=True,
            size_vertical=1,
        )
        ax.add_artist(bar)
        return ax


    def show_complex(
        complex_data,
        figax=None,
        vmin=None,
        vmax=None,
        power=None,
        ticks=True,
        chroma_boost=1,
        **kwargs,
    ):
        """ """
        rgb = Complex2RGB(
            complex_data, vmin, vmax, power=power, chroma_boost=chroma_boost
        )

        figsize = kwargs.pop("figsize", (6, 6))
        if figax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figax

        ax.imshow(rgb, **kwargs)
        if ticks is False:
            ax.set_xticks([])
            ax.set_yticks([])
        return ax
    return add_scalebar, show_complex


@app.cell
def _():
    # Imports
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from colorspacious import cspace_convert
    return AnchoredSizeBar, cspace_convert, mo, np, plt


@app.cell
def _():
    import typing as t
    import matplotlib.pyplot as pyplot
    import numpy as numpy
    from numpy.typing import ArrayLike, NDArray
    from matplotlib.colors import LinearSegmentedColormap, Normalize, Colormap
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib 




    class _FittedQDA:
        def __init__(self, means: ArrayLike, rotations: t.Sequence[ArrayLike],
                     scalings: t.Sequence[ArrayLike], priors: t.Optional[ArrayLike] = None):
            self.means_ = numpy.asarray(means)
            self.n_classes = self.means_.shape[0]
            self.rotations_ = list(map(numpy.asarray, rotations))
            self.scalings_ = list(map(numpy.asarray, scalings))
            self.priors_ = numpy.asarray(priors) if priors is not None else numpy.full(self.n_classes, 1/self.n_classes)

        def predict_log_proba(self, X: ArrayLike) -> NDArray[numpy.floating]:
            X = numpy.asarray(X)
            assert X.ndim == 2

            norm2 = []
            for i in range(self.n_classes):
                X2 = (X - self.means_[i]) @ (self.rotations_[i] * (self.scalings_[i] ** -0.5))
                norm2.append(numpy.sum(X2**2, axis=-1))

            norm2 = numpy.array(norm2).T
            u = numpy.asarray([numpy.sum(numpy.log(s)) for s in self.scalings_])
            scores = -0.5 * (norm2 + u) + numpy.log(self.priors_)

            log_likelihood = scores - numpy.max(scores, axis=-1, keepdims=True)
            log_likelihood -= numpy.log(numpy.sum(numpy.exp(log_likelihood), axis=-1, keepdims=True))

            return log_likelihood

        def predict_proba(self, X: ArrayLike) -> NDArray[numpy.floating]:
            return numpy.exp(self.predict_log_proba(X))

        def predict_prob_success(self, X: ArrayLike) -> NDArray[numpy.floating]:
            return numpy.exp(self.predict_log_proba(X)[..., 1])

    def _plot_probe_overlap(ax: 'Axes', metrics: t.Dict[str, float]):
        from matplotlib.patches import Circle, Rectangle

        probe_r = metrics['probe_radius']
        scan_r = metrics['scan_step'] / 2.0
        box_r = metrics['scan_step'] * metrics['fund_samp'] / 2.

        ax_r = scan_r + max(probe_r, box_r) * 1.1

        ax.set_xlim(-ax_r, ax_r)
        ax.set_ylim(-ax_r, ax_r)

        ax.set_axis_off()

        theta = 20.0 * numpy.pi / 180.

        pos1 = numpy.array([numpy.cos(theta), numpy.sin(theta)]) * scan_r

        _ = [ax.add_patch(Circle(
            t.cast(t.Tuple[float, float], tuple(pos)), probe_r, facecolor='green', alpha=0.6, edgecolor='black',
            linewidth=2.0,
        )) for pos in (pos1, -pos1)]

        ax.plot([pos1[0], -pos1[0]], [pos1[1], -pos1[1]], '.-k', linewidth=2.0)

        _ = [ax.add_patch(Rectangle(
            (pos[0] - box_r, pos[1] - box_r), 2*box_r, 2*box_r,
            edgecolor='black', linewidth=2.0, fill=False,
        )) for pos in (pos1, -pos1)]

    def predict_recons_success(ronchi_mag: ArrayLike, areal_oversamp: ArrayLike) -> NDArray[numpy.floating]:
        """
        Empirically predict the probability of reconstruction success, given the Ronchigram magnification
        and areal oversampling.

        Broadcasts `ronchi_mag` and `areal_oversamp` together, and returns an array of the same shape,
        with values indicating the estimated probability of success.

        Fitted on simulated Si data, using an intensity threshold of 90% to calculate the probe radius.
        """

        clf = _FittedQDA(
            means=[[0.49027803, 1.82918678], [0.80980859, 2.00158048]],
            rotations=[
                [[-0.2316652028331075, 0.972795576571098], [0.972795576571098, 0.2316652028331075]],
                [[-0.26985970170864165, 0.9628996528162853], [0.9628996528162853, 0.26985970170864165]]
            ],
            scalings=[[0.56440416, 0.07769331], [0.4110058 , 0.06621369]],
        )

        ronchi_mag, areal_oversamp = numpy.broadcast_arrays(ronchi_mag, areal_oversamp)

        return clf.predict_prob_success(
            numpy.log10(numpy.stack((ronchi_mag, areal_oversamp), axis=-1).reshape(-1, 2))
        ).reshape(ronchi_mag.shape)

    def _plot_predicted_success(ax: 'Axes', metrics: t.Dict[str, float]):


        bwr_r: Colormap = matplotlib.colormaps['bwr_r']  # type: ignore

        gamma = 2.0
        lin_norm = Normalize(0.5, 1.0)
        prob_cmap = LinearSegmentedColormap.from_list(
            'prob', numpy.stack([
                bwr_r(lin_norm.inverse(numpy.abs(lin_norm(x))**gamma * numpy.sign(lin_norm(x))))
                for x in numpy.linspace(0., 1., 512, endpoint=True)
            ], axis=0)
        )

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel("Ronchi. mag [$\\mathrm{px/\\AA}$]")
        ax.set_ylabel("Areal oversampling")

        ronchi_mag = metrics['ronchi_mag']
        areal_oversamp = metrics['areal_oversamp']

        ylim = (
            min(8e-1, areal_oversamp / 3),
            max(1e4, areal_oversamp * 3),
        )
        xlim = (
            min(8e-1, ronchi_mag / 1.5),
            max(1e2, ronchi_mag * 4),
        )

        yy = numpy.geomspace(ylim[1], ylim[0], 100)
        xx = numpy.geomspace(xlim[0], xlim[1], 100)
        yy, xx = numpy.meshgrid(yy, xx, indexing='ij')

        pp = predict_recons_success(xx, yy)
        ax.pcolormesh(xx, yy, pp, alpha=0.8, cmap=prob_cmap, vmin=0.0, vmax=1.0)

        prob = predict_recons_success(ronchi_mag, areal_oversamp)
        ax.scatter([ronchi_mag], [areal_oversamp], marker='x', s=80, c='black')
        ax.annotate(f"{prob:.1%}", (ronchi_mag, areal_oversamp),
                    (0.5, 0.7), textcoords='offset fontsize')

    def _plot_linear_metrics(ax: 'Axes', metrics: t.Dict[str, float]):
        # mrad
        diff_scale = metrics['wavelength']*1e3/(2. * metrics['probe_radius'])
        step_scale = 2.*metrics['probe_radius']

        x = metrics['scan_step']
        y = metrics['diff_step']

        ax.set_xlabel("Scan step [$\\mathrm{\\AA/px}$]")
        ax.set_xscale('log')
        ax.set_ylabel("Diff. pixel size [mrad/px]")
        ax.set_yscale('log')

        ylim = (
            min(y/1.5, 0.1*diff_scale),
            max(y*1.5, 10.*diff_scale),
        )
        xlim = (
            min(x/1.5, 0.1*step_scale),
            max(x*1.5, 10.*step_scale),
        )
        yy = numpy.geomspace(*ylim, 201, endpoint=True)
        xx = numpy.geomspace(*xlim, 201, endpoint=True)
        (yy, xx) = numpy.meshgrid(yy, xx, indexing='ij')

        fps = 1.0 / (yy / diff_scale * xx / step_scale)

        ax.axhline(diff_scale, linestyle='dashed', color='#4363d8') # blue
        ax.axvline(step_scale, linestyle='dashdot', color='#f58231') # orange
        ax.contour(xx, yy, fps, [1.0], linestyles=['solid'], colors=['#e6194b']) # red

        ax.scatter([x], [y], marker='x', color='black', s=80)

        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)

    def plot_metrics(metrics: t.Dict[str, float]) -> 'Figure':
        fig, (ax1, ax2, ax3) = pyplot.subplots(
            ncols=3,
            gridspec_kw={
                'width_ratios': [2., 1., 2.],
                'wspace': 0.05,
            },
            constrained_layout=True
        )
        fig.set_size_inches(12, 4)
        ax1.set_box_aspect(1.)
        ax2.set_aspect(1.)
        ax3.set_box_aspect(1.)
        ax1.set_title('Probe sampling vs. Diff. Oversampling')
        ax2.set_title('Probe Overlap')
        ax3.set_title('Heuristic success probability')

        _plot_linear_metrics(ax1, metrics)
        _plot_probe_overlap(ax2, metrics)
        _plot_predicted_success(ax3, metrics)

        return fig





    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
