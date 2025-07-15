import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
 
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.fft import fft2, ifft2, fftshift, ifftshift

    from collections.abc import Sequence
    from typing import Any, List, Optional, Tuple, Union, cast

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
 
    from numpy.typing import NDArray
    from dataclasses import dataclass
    from typing import Any, List, Optional, Sequence, Tuple, Union, cast

    import matplotlib as mpl
 
    from colorspacious import cspace_convert
    from matplotlib import cm, colors, legend, ticker
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from numpy.typing import NDArray
 

    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from typing import Any

    import numpy as np
    from matplotlib import colors
    from numpy.typing import NDArray

    """
    Custom normalization based on astropy's visualization routines.

    Original implementation:
    https://github.com/astropy/astropy/blob/main/astropy/visualization/mpl_normalize.py

    Licensed under a 3-clause BSD style license.
    """


    class BaseInterval(ABC):
        """
        Base class for the interval classes, which when called with an array of values,
        return an interval clipped to the [0:1] range.
        """

        @abstractmethod
        def get_limits(self, values: NDArray) -> tuple[float, float]:
            """
            Get the minimum and maximum values for the interval.
            This method must be implemented by subclasses.

            Parameters
            ----------
            values : array-like
                The input values.

            Returns
            -------
            vmin, vmax : float
                The minimum and maximum values.
            """
            raise NotImplementedError("Subclasses must implement get_limits")

        def __call__(self, values: NDArray) -> NDArray:
            """
            Transform values using this interval.

            Parameters
            ----------
            values : array-like
                The input values.

            Returns
            -------
            result : ndarray
                The transformed values.
            """
            vmin, vmax = self.get_limits(values)

            # subtract vmin
            values = np.subtract(values, vmin)

            # divide by interval
            if (vmax - vmin) != 0.0:
                np.true_divide(values, vmax - vmin, out=values)

            # clip to [0:1]
            np.clip(values, 0.0, 1.0, out=values)
            return values

        def inverse(self, values: NDArray) -> NDArray:
            """
            Pseudo-inverse interval transform. Note this does not recover
            the original range due to clipping. Used for colorbars.

            Parameters
            ----------
            values : array-like
                The input values.

            Returns
            -------
            result : ndarray
                The transformed values.
            """
            vmin, vmax = self.get_limits(values)

            values = np.multiply(values, vmax - vmin)
            np.add(values, vmin, out=values)
            return values


    @dataclass
    class ManualInterval(BaseInterval):
        """
        Interval based on user-specified values.

        Parameters
        ----------
        vmin : float, optional
            The minimum value in the scaling.
        vmax : float, optional
            The maximum value in the scaling.
        """

        vmin: float | None = None
        vmax: float | None = None

        def get_limits(self, values: NDArray) -> tuple[float, float]:
            # Avoid overhead of preparing array if both limits have been specified
            # manually, for performance.

            if self.vmin is not None and self.vmax is not None:
                return self.vmin, self.vmax

            # Make sure values is a Numpy array
            values = np.asarray(values).ravel()

            # Filter out invalid values (inf, nan)
            values = values[np.isfinite(values)]
            vmin = np.min(values) if self.vmin is None else self.vmin
            vmax = np.max(values) if self.vmax is None else self.vmax

            return vmin, vmax


    @dataclass
    class CenteredInterval(BaseInterval):
        """
        Centered interval based on user-specified halfrange.

        Parameters
        ----------
        vcenter : float
            The center value in the scaling.
        half_range : float, optional
            The half range in the scaling.
        """

        vcenter: float = 0.0
        half_range: float | None = None

        def get_limits(self, values: NDArray) -> tuple[float, float]:
            if self.half_range is not None:
                return self.vcenter - self.half_range, self.vcenter + self.half_range

            values = np.asarray(values).ravel()
            values = values[np.isfinite(values)]
            vmin = np.min(values)
            vmax = np.max(values)

            half_range = np.maximum(
                np.abs(vmin - self.vcenter), np.abs(vmax - self.vcenter)
            )

            return self.vcenter - half_range, self.vcenter + half_range


    @dataclass
    class QuantileInterval(BaseInterval):
        """
        Interval based on a keeping a specified fraction of pixels.

        Parameters
        ----------
        lower_quantile : float or None
            The lower quantile below which to ignore pixels. If None, then
            defaults to 0.
        upper_quantile : float or None
            The upper quantile above which to ignore pixels. If None, then
            defaults to 1.
        """

        lower_quantile: float = 0.02
        upper_quantile: float = 0.98

        def get_limits(self, values: NDArray) -> tuple[float, float]:
            # Make sure values is a Numpy array
            values = np.asarray(values).ravel()

            # Filter out invalid values (inf, nan)
            values = values[np.isfinite(values)]
            vmin, vmax = np.quantile(values, (self.lower_quantile, self.upper_quantile))  # type: ignore

            return vmin, vmax


    @dataclass
    class LinearStretch:
        r"""
        A linear stretch with a slope and offset.

        The stretch is given by:

        .. math::
            y = slope * x + intercept

        Parameters
        ----------
        slope : float, optional
            The ``slope`` parameter used in the above formula.  Default is 1.
        intercept : float, optional
            The ``intercept`` parameter used in the above formula.  Default is 0.
        """

        slope: float = 1.0
        intercept: float = 0.0

        def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
            if self.slope == 1.0 and self.intercept == 0.0:
                return values

            values = np.array(values, copy=copy)
            np.clip(values, 0.0, 1.0, out=values)
            if self.slope != 1.0:
                np.multiply(values, self.slope, out=values)
            if self.intercept != 0.0:
                np.add(values, self.intercept, out=values)
            return values

        @property
        def inverse(self) -> "LinearStretch":
            return LinearStretch(1 / self.slope, -self.intercept / self.slope)


    @dataclass
    class PowerLawStretch:
        r"""
        A power stretch.

        The stretch is given by:

        .. math::
            y = x^{power}

        Parameters
        ----------
        power : float
            The power index (see the above formula).  ``power`` must be greater
            than 0.
        """

        power: float = 1.0

        def __post_init__(self) -> None:
            if self.power <= 0.0:
                raise ValueError("power must be > 0")

        def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
            if self.power == 1.0:
                return values

            values = np.array(values, copy=copy)
            np.clip(values, 0.0, 1.0, out=values)
            np.power(values, self.power, out=values)
            return values

        @property
        def inverse(self) -> "PowerLawStretch":
            return PowerLawStretch(1.0 / self.power)


    @dataclass
    class LogarithmicStretch:
        r"""
        A logarithmic stretch.

        The stretch is given by:

        .. math::
            y = \frac{\log{(a x + 1)}}{\log{(a + 1)}}

        Parameters
        ----------
        a : float
            The ``a`` parameter used in the above formula.  ``a`` must be
            greater than 0.  Default is 1000.
        """

        a: float = 1000.0

        def __post_init__(self) -> None:
            if self.a <= 0:
                raise ValueError("a must be > 0")

        def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
            values = np.array(values, copy=copy)
            np.clip(values, 0.0, 1.0, out=values)
            np.multiply(values, self.a, out=values)
            np.add(values, 1.0, out=values)
            np.log(values, out=values)
            np.true_divide(values, np.log(self.a + 1.0), out=values)
            return values

        @property
        def inverse(self) -> "InverseLogarithmicStretch":
            return InverseLogarithmicStretch(self.a)


    @dataclass
    class InverseLogarithmicStretch:
        r"""
        Inverse transformation for `LogarithmicStretch`.

        The stretch is given by:

        .. math::
            y = \frac{e^{y \log{a + 1}} - 1}{a} \\
            y = \frac{e^{y} (a + 1) - 1}{a}

        Parameters
        ----------
        a : float, optional
            The ``a`` parameter used in the above formula.  ``a`` must be
            greater than 0.  Default is 1000.
        """

        a: float = 1000.0

        def __post_init__(self) -> None:
            if self.a <= 0:
                raise ValueError("a must be > 0")

        def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
            values = np.array(values, copy=copy)
            np.clip(values, 0.0, 1.0, out=values)
            np.multiply(values, np.log(self.a + 1.0), out=values)
            np.exp(values, out=values)
            np.subtract(values, 1.0, out=values)
            np.true_divide(values, self.a, out=values)
            return values

        @property
        def inverse(self) -> "LogarithmicStretch":
            return LogarithmicStretch(self.a)


    @dataclass
    class InverseHyperbolicSineStretch:
        r"""
        An asinh stretch.

        The stretch is given by:

        .. math::
            y = \frac{{\rm asinh}(x / a)}{{\rm asinh}(1 / a)}.

        Parameters
        ----------
        a : float, optional
            The ``a`` parameter used in the above formula. The value of this
            parameter is where the asinh curve transitions from linear to
            logarithmic behavior, expressed as a fraction of the normalized
            image. The stretch becomes more linear as the ``a`` value is
            increased. ``a`` must be greater than 0. Default is 0.1.
        """

        a: float = 0.1

        def __post_init__(self) -> None:
            if self.a <= 0:
                raise ValueError("a must be > 0")

        def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
            values = np.array(values, copy=copy)
            np.clip(values, 0.0, 1.0, out=values)
            # map to [-1,1]
            np.multiply(values, 2.0, out=values)
            np.subtract(values, 1.0, out=values)

            np.true_divide(values, self.a, out=values)
            np.arcsinh(values, out=values)

            # map from [-1,1]
            np.true_divide(values, np.arcsinh(1.0 / self.a) * 2.0, out=values)
            np.add(values, 0.5, out=values)
            return values

        @property
        def inverse(self) -> "HyperbolicSineStretch":
            return HyperbolicSineStretch(1.0 / np.arcsinh(1.0 / self.a))


    @dataclass
    class HyperbolicSineStretch:
        r"""
        A sinh stretch.

        The stretch is given by:

        .. math::
            y = \frac{{\rm sinh}(x / a)}{{\rm sinh}(1 / a)}

        Parameters
        ----------
        a : float, optional
            The ``a`` parameter used in the above formula. The stretch
            becomes more linear as the ``a`` value is increased. ``a`` must
            be greater than 0. Default is 1/3.
        """

        a: float = 1.0 / 3.0

        def __post_init__(self) -> None:
            if self.a <= 0:
                raise ValueError("a must be > 0")

        def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
            values = np.array(values, copy=copy)
            np.clip(values, 0.0, 1.0, out=values)

            # map to [-1,1]
            np.subtract(values, 0.5, out=values)
            np.multiply(values, 2.0, out=values)

            np.true_divide(values, self.a, out=values)
            np.sinh(values, out=values)

            # map from [-1,1]
            np.true_divide(values, np.sinh(1.0 / self.a) * 2.0, out=values)
            np.add(values, 0.5, out=values)
            return values

        @property
        def inverse(self) -> "InverseHyperbolicSineStretch":
            return InverseHyperbolicSineStretch(1.0 / np.sinh(1.0 / self.a))


    class CustomNormalization(colors.Normalize):
        """A flexible normalization class that combines interval and stretch operations.

        This class extends matplotlib's Normalize class to provide more sophisticated
        normalization options for visualization. It combines an interval operation
        (which maps data to a [0,1] range) with a stretch operation (which applies
        a transformation to the normalized data).

        Parameters
        ----------
        interval_type : str, default="quantile"
            Type of interval to use. Options are "quantile", "manual", or "centered".
        stretch_type : str, default="linear"
            Type of stretch to apply. Options are "linear", "power", "logarithmic", or "asinh".
        data : ndarray, optional
            Data array to use for setting limits if not explicitly provided.
        lower_quantile : float, default=0.02
            Lower quantile for "quantile" interval type.
        upper_quantile : float, default=0.98
            Upper quantile for "quantile" interval type.
        vmin : float, optional
            Minimum value for "manual" interval type.
        vmax : float, optional
            Maximum value for "manual" interval type.
        vcenter : float, default=0.0
            Center value for "centered" interval type.
        half_range : float, optional
            Half range for "centered" interval type.
        power : float, default=1.0
            Power for "power" stretch type.
        logarithmic_index : float, default=1000.0
            Index for "logarithmic" stretch type.
        asinh_linear_range : float, default=0.1
            Linear range for "asinh" stretch type.
        """

        def __init__(
            self,
            interval_type: str = "quantile",
            stretch_type: str = "linear",
            *,
            data: NDArray | None = None,
            lower_quantile: float = 0.02,
            upper_quantile: float = 0.98,
            vmin: float | None = None,
            vmax: float | None = None,
            vcenter: float = 0.0,
            half_range: float | None = None,
            power: float = 1.0,
            logarithmic_index: float = 1000.0,
            asinh_linear_range: float = 0.1,
        ) -> None:
            """Initialize the CustomNormalization object."""
            super().__init__(vmin=vmin, vmax=vmax, clip=False)
            if interval_type == "quantile":
                self.interval = QuantileInterval(
                    lower_quantile=lower_quantile, upper_quantile=upper_quantile
                )
            elif interval_type == "manual":
                self.interval = ManualInterval(vmin=vmin, vmax=vmax)
            elif interval_type == "centered":
                self.interval = CenteredInterval(
                    vcenter=vcenter,
                    half_range=half_range,
                )
            else:
                raise ValueError("unrecognized interval_type.")

            if stretch_type == "power" or power != 1.0:
                self.stretch = PowerLawStretch(power)
            elif stretch_type == "linear":
                self.stretch = LinearStretch()
            elif stretch_type == "logarithmic":
                self.stretch = LogarithmicStretch(logarithmic_index)
            elif stretch_type == "asinh":
                self.stretch = InverseHyperbolicSineStretch(asinh_linear_range)
            else:
                raise ValueError("unrecognized stretch_type.")

            self.vmin = vmin
            self.vmax = vmax

            if data is not None:
                self._set_limits(data)

        def _set_limits(self, data: NDArray) -> None:
            """Set the normalization limits based on the provided data.

            Parameters
            ----------
            data : ndarray
                The data array to use for setting limits.

            Returns
            -------
            None
            """
            self.vmin, self.vmax = self.interval.get_limits(data)
            self.interval = ManualInterval(
                self.vmin, self.vmax
            )  # set explicitly with ManualInterval
            return None

        def __call__(self, value: NDArray, clip: bool | None = None) -> NDArray:  # type: ignore[override]
            """Apply the normalization to the input values.

            Parameters
            ----------
            value : array-like
                The input values to normalize.
            clip : bool, optional
                If True, clip the normalized values to [0, 1].

            Returns
            -------
            ndarray
                The normalized values, with invalid values masked.
            """
            values = self.interval(value)
            self.stretch(values, copy=False)
            return np.ma.masked_invalid(values)

        def inverse(self, value: NDArray) -> NDArray:  # type: ignore[override]
            """Apply the inverse normalization to the input values.

            Parameters
            ----------
            value : array-like
                The input values to inverse normalize.

            Returns
            -------
            ndarray
                The inverse normalized values.
            """
            values = self.stretch.inverse(value)
            values = self.interval.inverse(values)
            return values


    @dataclass
    class NormalizationConfig:
        """Configuration for CustomNormalization.

        This dataclass provides a convenient way to specify normalization parameters
        for the CustomNormalization class.

        Parameters
        ----------
        interval_type : str, default="quantile"
            Type of interval to use. Options are "quantile", "manual", or "centered".
        stretch_type : str, default="linear"
            Type of stretch to apply. Options are "linear", "power", "logarithmic", or "asinh".
        lower_quantile : float, default=0.02
            Lower quantile for "quantile" interval type.
        upper_quantile : float, default=0.98
            Upper quantile for "quantile" interval type.
        vmin : float, optional
            Minimum value for "manual" interval type.
        vmax : float, optional
            Maximum value for "manual" interval type.
        vcenter : float, default=0.0
            Center value for "centered" interval type.
        half_range : float, optional
            Half range for "centered" interval type.
        power : float, default=1.0
            Power for "power" stretch type.
        logarithmic_index : float, default=1000.0
            Index for "logarithmic" stretch type.
        asinh_linear_range : float, default=0.1
            Linear range for "asinh" stretch type.
        """

        interval_type: str = "quantile"
        stretch_type: str = "linear"
        lower_quantile: float = 0.02
        upper_quantile: float = 0.98
        vmin: float | None = None
        vmax: float | None = None
        vcenter: float = 0.0
        half_range: float | None = None
        power: float = 1.0
        logarithmic_index: float = 1000.0
        asinh_linear_range: float = 0.1


    NORMALIZATION_PRESETS = {
        "linear_auto": lambda: NormalizationConfig(),
        "linear_minmax": lambda: NormalizationConfig(interval_type="manual"),
        "linear_centered": lambda: NormalizationConfig(interval_type="centered"),
        "log_auto": lambda: NormalizationConfig(stretch_type="logarithmic"),
        "log_minmax": lambda: NormalizationConfig(
            stretch_type="logarithmic", interval_type="manual"
        ),
        "power_squared": lambda: NormalizationConfig(stretch_type="power", power=2.0),
        "power_sqrt": lambda: NormalizationConfig(stretch_type="power", power=0.5),
        "asinh_centered": lambda: NormalizationConfig(
            stretch_type="asinh", interval_type="centered"
        ),
    }


    def _resolve_normalization(norm: Any) -> NormalizationConfig:
        """Resolve various input types to a NormalizationConfig object.

        This function takes different input types and converts them to a
        NormalizationConfig object that can be used with CustomNormalization.

        Parameters
        ----------
        norm : None or dict or str or NormalizationConfig
            The normalization configuration to resolve.

        Returns
        -------
        NormalizationConfig
            The resolved normalization configuration.

        Raises
        ------
        ValueError
            If norm is a string that doesn't match any preset.
        TypeError
            If norm is not one of the supported types.
        """
        if norm is None:
            return NormalizationConfig()
        elif isinstance(norm, dict):
            return NormalizationConfig(**norm)
        elif isinstance(norm, str):
            if norm not in NORMALIZATION_PRESETS:
                raise ValueError(f"Unknown normalization preset: {norm}")
            return NORMALIZATION_PRESETS[norm]()
        elif isinstance(norm, NormalizationConfig):
            return norm
        else:
            raise TypeError("norm must be None, dict, str, or NormalizationConfig")



    def array_to_rgba(
        scaled_amplitude: NDArray,
        scaled_angle: Optional[NDArray] = None,
        *,
        cmap: Union[str, colors.Colormap] = "gray",
        chroma_boost: float = 1,
    ) -> NDArray:
        """Convert amplitude and angle arrays to an RGBA color array.

        This function creates a color representation of data using either a simple colormap
        or a perceptually-uniform color space based on amplitude and angle information.

        Parameters
        ----------
        scaled_amplitude : np.ndarray
            Array of amplitude values, typically normalized to [0, 1].
        scaled_angle : np.ndarray, optional
            Array of angle values in radians. If provided, creates a color representation
            using the JCh color space where amplitude controls lightness and angle controls hue.
        cmap : str or mpl.colors.Colormap, default="gray"
            Colormap to use when scaled_angle is None.
        chroma_boost : float, default=1
            Factor to boost color saturation when using angle-based coloring.

        Returns
        -------
        np.ndarray
            RGBA array with shape (height, width, 4) where the last dimension contains
            (red, green, blue, alpha) values in the range [0, 1].

        Raises
        ------
        ValueError
            If scaled_angle is provided but has a different shape than scaled_amplitude.
        """
        cmap_obj = (
            cmap if isinstance(cmap, colors.Colormap) else mpl.colormaps.get_cmap(cmap)
        )
        if scaled_angle is None:
            rgba = cmap_obj(scaled_amplitude)
        else:
            if scaled_angle.shape != scaled_amplitude.shape:
                raise ValueError()

            J = scaled_amplitude * 61.5
            C = np.minimum(chroma_boost * 98 * J / 123, 110)
            h = np.rad2deg(scaled_angle) + 180

            JCh = np.stack((J, C, h), axis=-1)
            with np.errstate(invalid="ignore"):
                rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

            alpha = np.ones_like(scaled_amplitude)
            rgba = np.dstack((rgb, alpha))

        return rgba


    def list_of_arrays_to_rgba(
        list_of_arrays: List[NDArray],
        *,
        norm: CustomNormalization = CustomNormalization(),
        chroma_boost: float = 1,
    ) -> NDArray:
        """Converts a list of arrays to a perceptually-uniform RGB array.

        This function takes multiple arrays and creates a color representation where each
        array is assigned a unique hue angle, and the amplitude of each array determines
        the contribution to the final color. The result is a perceptually-uniform color
        representation that can effectively visualize multiple data sources simultaneously.

        Parameters
        ----------
        list_of_arrays : list of np.ndarray
            List of arrays to be converted to a color representation. All arrays must have
            the same shape.
        norm : CustomNormalization, default=CustomNormalization()
            Normalization to apply to each array before processing.
        chroma_boost : float, default=1
            Factor to boost color saturation in the final output.

        Returns
        -------
        np.ndarray
            RGBA array with shape (height, width, 4) representing the combined data.
        """
        list_of_arrays = [norm(array) for array in list_of_arrays]
        bins = np.asarray(list_of_arrays)
        n = bins.shape[0]

        # circular encoding
        hue_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        hue_angles += np.linspace(0.0, 0.5, n) * (
            2 * np.pi / n / 2
        )  # jitter to avoid cancellation
        complex_weights = np.exp(1j * hue_angles)[:, None, None] * bins

        # weighted average direction (w/ normalization)
        complex_sum = complex_weights.sum(0)
        scaled_amplitude = np.clip(np.abs(complex_sum), 0, 1)
        scaled_angle = np.angle(complex_sum)

        return array_to_rgba(scaled_amplitude, scaled_angle, chroma_boost=chroma_boost)


    @dataclass
    class ScalebarConfig:
        """Configuration for adding a scale bar to a plot.

        Attributes
        ----------
        sampling : float, default=1.0
            Physical units per pixel.
        units : str, default="pixels"
            Units to display on the scale bar.
        length : float, optional
            Length of the scale bar in physical units. If None, an appropriate length
            will be estimated.
        width_px : float, default=1
            Width of the scale bar in pixels.
        pad_px : float, default=0.5
            Padding around the scale bar in pixels.
        color : str, default="white"
            Color of the scale bar.
        loc : str or int, default="lower right"
            Location of the scale bar on the plot. Can be a string like "lower right"
            or an integer location code.
        """

        sampling: float = 1.0
        units: str = "pixels"
        length: Optional[float] = None
        width_px: float = 1
        pad_px: float = 0.5
        color: str = "white"
        loc: Union[str, int] = "lower right"


    def _resolve_scalebar(cfg: Any) -> Optional[ScalebarConfig]:
        """Resolve various input types to a ScalebarConfig object.

        Parameters
        ----------
        cfg : None, bool, dict, or ScalebarConfig
            Configuration for the scale bar.

        Returns
        -------
        ScalebarConfig or None
            Resolved configuration object or None if cfg is None or False.

        Raises
        ------
        TypeError
            If cfg is not one of the supported types.
        """
        if cfg is None or cfg is False:
            return None
        elif cfg is True:
            return ScalebarConfig()
        elif isinstance(cfg, dict):
            return ScalebarConfig(**cfg)
        elif isinstance(cfg, ScalebarConfig):
            return cfg
        else:
            raise TypeError("scalebar must be None, dict, bool, or ScalebarConfig")


    def estimate_scalebar_length(length: float, sampling: float) -> Tuple[float, float]:
        """Estimate an appropriate scale bar length based on data dimensions.

        This function calculates a "nice" scale bar length that is a multiple of
        0.5, 1.0, or 2.0 times a power of 10, depending on the data range.

        Parameters
        ----------
        length : float
            Total length of the data in physical units.
        sampling : float
            Physical units per pixel.

        Returns
        -------
        tuple
            (length_units, length_pixels) where length_units is the estimated
            scale bar length in physical units and length_pixels is the equivalent
            in pixels.
        """
        d = length * sampling / 2
        exp = np.floor(np.log10(d))
        base = d / (10**exp)
        if base >= 1 and base < 2.1:
            spacing = 0.5
        elif base >= 2.1 and base < 4.6:
            spacing = 1.0
        elif base >= 4.6 and base <= 10:
            spacing = 2.0
        else:
            spacing = 1.0  # default case
        spacing = spacing * 10**exp
        return spacing, spacing / sampling


    def add_scalebar_to_ax(
        ax: Axes,
        array_size: float,
        sampling: float,
        length_units: Optional[float],
        units: str,
        width_px: float,
        pad_px: float,
        color: str,
        loc: Union[str, int],
    ) -> None:
        """Add a scale bar to a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to add the scale bar to.
        array_size : float
            Size of the data array in pixels.
        sampling : float
            Physical units per pixel.
        length_units : float, optional
            Length of the scale bar in physical units. If None, an appropriate length
            will be estimated.
        units : str
            Units to display on the scale bar.
        width_px : float
            Width of the scale bar in pixels.
        pad_px : float
            Padding around the scale bar in pixels.
        color : str
            Color of the scale bar.
        loc : str or int
            Location of the scale bar on the plot.
        """
        if length_units is None:
            length_units, length_px = estimate_scalebar_length(array_size, sampling)
        else:
            length_px = length_units / sampling

        if length_units % 1 == 0.0:
            label = f"{length_units:.0f} {units}"
        else:
            label = f"{length_units:.2f} {units}"

        if isinstance(loc, int):
            loc_codes = legend.Legend.codes
            loc_strings = {v: k for k, v in loc_codes.items()}
            loc = loc_strings[loc]

        bar = AnchoredSizeBar(
            ax.transData,
            length_px,
            label,
            loc,
            pad=pad_px,
            color=color,
            frameon=False,
            label_top=loc[:3] == "low",
            size_vertical=int(width_px),  # Convert to int as required by AnchoredSizeBar
        )
        ax.add_artist(bar)


    def add_cbar_to_ax(
        fig: Figure,
        cax: Axes,
        norm: colors.Normalize,
        cmap: colors.Colormap,
        eps: float = 1e-8,
    ) -> Colorbar:
        """Add a colorbar to a matplotlib figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to add the colorbar to.
        cax : matplotlib.axes.Axes
            The axes to place the colorbar in.
        norm : matplotlib.colors.Normalize
            The normalization for the colormap.
        cmap : matplotlib.colors.Colormap
            The colormap to use.
        eps : float, default=1e-8
            Small value to avoid floating point errors when filtering ticks.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The created colorbar object.
        """
        tick_locator = ticker.AutoLocator()
        vmin = cast(float, norm.vmin)  # Cast to float since we know it can't be None
        vmax = cast(float, norm.vmax)  # Cast to float since we know it can't be None
        ticks = tick_locator.tick_values(vmin, vmax)
        # Convert to numpy array for boolean indexing
        ticks_arr = np.asarray(ticks)
        mask = (ticks_arr >= vmin - eps) & (ticks_arr <= vmax + eps)
        ticks = ticks_arr[mask]

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, ticks=ticks, format=formatter)
        return cb


    def add_arg_cbar_to_ax(
        fig: Figure,
        cax: Axes,
        chroma_boost: float = 1,
    ) -> Colorbar:
        """Add a colorbar for phase values to a matplotlib figure.

        This function creates a colorbar suitable for displaying phase values.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to add the colorbar to.
        cax : matplotlib.axes.Axes
            The axes to place the colorbar in.
        chroma_boost : float, default=1
            Factor to boost color saturation.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The created colorbar object.
        """
        h = np.linspace(0, 360, 256, endpoint=False)
        J = np.full_like(h, 61.5)
        C = np.full_like(h, np.minimum(49 * chroma_boost, 110))
        JCh = np.stack((J, C, h), axis=-1)
        rgb_vals = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

        angle_cmap = colors.ListedColormap(rgb_vals)
        angle_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        sm = cm.ScalarMappable(norm=angle_norm, cmap=angle_cmap)
        cb_angle = fig.colorbar(sm, cax=cax)

        cb_angle.set_label("arg", rotation=0, ha="center", va="bottom")
        cb_angle.ax.yaxis.set_label_coords(0.5, -0.05)
        cb_angle.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb_angle.set_ticklabels(
            [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
        )

        return cb_angle


 

 
    from custom_normalizations import (
        CustomNormalization,
        NormalizationConfig,
        _resolve_normalization,
    )
    from visualization_utils import (
        ScalebarConfig,
        _resolve_scalebar,
        add_arg_cbar_to_ax,
        add_cbar_to_ax,
        add_scalebar_to_ax,
        array_to_rgba,
        list_of_arrays_to_rgba,
    )


    def _show_2d(
        array: NDArray,
        *,
        norm: Optional[Union[NormalizationConfig, dict, str]] = None,
        scalebar: Optional[Union[ScalebarConfig, dict, bool]] = None,
        cmap: Union[str, colors.Colormap] = "gray",
        chroma_boost: float = 1.0,
        cbar: bool = False,
        figax: Optional[Tuple[Any, Any]] = None,
        figsize: Tuple[int, int] = (8, 8),
        title: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """Display a 2D array as an image with optional colorbar and scalebar.

        This function visualizes a 2D array, handling both real and complex data.
        For complex data, it displays amplitude and phase information using a
        perceptually-uniform color representation.

        Parameters
        ----------
        array : ndarray
            The 2D array to visualize. Can be real or complex.
        norm : NormalizationConfig or dict or str, optional
            Configuration for normalizing the data before visualization.
        scalebar : ScalebarConfig or dict or bool, optional
            Configuration for adding a scale bar to the plot.
        cmap : str or Colormap, default="gray"
            Colormap to use for real data or amplitude of complex data.
        chroma_boost : float, default=1.0
            Factor to boost color saturation when displaying complex data.
        cbar : bool, default=False
            Whether to add a colorbar to the plot.
        figax : tuple, optional
            (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
        figsize : tuple, default=(8, 8)
            Figure size in inches, used only if figax is None.
        title : str, optional
            Title for the plot.

        Returns
        -------
        fig : Figure
            The matplotlib figure object.
        ax : Axes
            The matplotlib axes object.
        """
        is_complex = np.iscomplexobj(array)
        if is_complex:
            amplitude = np.abs(array)
            angle = np.angle(array)
        else:
            amplitude = array
            angle = None

        norm_config = _resolve_normalization(norm)
        scalebar_config = _resolve_scalebar(scalebar)

        norm_obj = CustomNormalization(
            interval_type=norm_config.interval_type,
            stretch_type=norm_config.stretch_type,
            lower_quantile=norm_config.lower_quantile,
            upper_quantile=norm_config.upper_quantile,
            vmin=norm_config.vmin,
            vmax=norm_config.vmax,
            vcenter=norm_config.vcenter,
            half_range=norm_config.half_range,
            power=norm_config.power,
            logarithmic_index=norm_config.logarithmic_index,
            asinh_linear_range=norm_config.asinh_linear_range,
            data=amplitude,
        )

        scaled_amplitude = norm_obj(amplitude)
        rgba = array_to_rgba(scaled_amplitude, angle, cmap=cmap, chroma_boost=chroma_boost)

        if figax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figax

        ax.imshow(rgba)
        ax.set(xticks=[], yticks=[], title=title)

        if cbar:
            divider = make_axes_locatable(ax)
            ax_cb_abs = divider.append_axes("right", size="5%", pad="2.5%")
            # Convert cmap to Colormap if it's a string
            cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
            cb_abs = add_cbar_to_ax(fig, ax_cb_abs, norm_obj, cmap_obj)

            if is_complex:
                ax_cb_angle = divider.append_axes("right", size="5%", pad="10%")
                add_arg_cbar_to_ax(fig, ax_cb_angle, chroma_boost=chroma_boost)
                cb_abs.set_label("abs", rotation=0, ha="center", va="bottom")
                cb_abs.ax.yaxis.set_label_coords(0.5, -0.05)

        if scalebar_config is not None:
            add_scalebar_to_ax(
                ax,
                rgba.shape[1],
                scalebar_config.sampling,
                scalebar_config.length,
                scalebar_config.units,
                scalebar_config.width_px,
                scalebar_config.pad_px,
                scalebar_config.color,
                scalebar_config.loc,
            )

        return fig, ax


    def _show_2d_combined(
        list_of_arrays: Sequence[NDArray],
        *,
        norm: Optional[Union[NormalizationConfig, dict, str]] = None,
        scalebar: Optional[Union[ScalebarConfig, dict, bool]] = None,
        cmap: Union[str, colors.Colormap] = "gray",
        chroma_boost: float = 1.0,
        cbar: bool = False,
        figax: Optional[Tuple[Any, Any]] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> Tuple[Any, Any]:
        """Display multiple 2D arrays as a single combined image.

        This function takes a list of 2D arrays and creates a single visualization
        where each array is assigned a unique color, and their amplitudes determine
        the contribution to the final color. This is useful for comparing multiple
        related datasets.

        Parameters
        ----------
        list_of_arrays : sequence of ndarray
            Sequence of 2D arrays to combine into a single visualization.
        norm : NormalizationConfig or dict or str, optional
            Configuration for normalizing the data before visualization.
        scalebar : ScalebarConfig or dict or bool, optional
            Configuration for adding a scale bar to the plot.
        cmap : str or Colormap, default="gray"
            Base colormap to use (though each array will get a unique color).
        chroma_boost : float, default=1.0
            Factor to boost color saturation.
        cbar : bool, default=False
            Whether to add a colorbar to the plot (not yet implemented).
        figax : tuple, optional
            (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
        figsize : tuple, default=(8, 8)
            Figure size in inches, used only if figax is None.

        Returns
        -------
        fig : Figure
            The matplotlib figure object.
        ax : Axes
            The matplotlib axes object.

        Raises
        ------
        NotImplementedError
            If cbar is True (colorbar for combined visualization not yet implemented).
        """
        norm_config = _resolve_normalization(norm)
        scalebar_config = _resolve_scalebar(scalebar)

        norm_obj = CustomNormalization(
            interval_type=norm_config.interval_type,
            stretch_type=norm_config.stretch_type,
            lower_quantile=norm_config.lower_quantile,
            upper_quantile=norm_config.upper_quantile,
            vmin=norm_config.vmin,
            vmax=norm_config.vmin,
            vcenter=norm_config.vcenter,
            half_range=norm_config.half_range,
            power=norm_config.power,
            logarithmic_index=norm_config.logarithmic_index,
            asinh_linear_range=norm_config.asinh_linear_range,
        )

        # Convert Sequence to List for list_of_arrays_to_rgba
        list_of_arrays_list = list(list_of_arrays)
        rgba = list_of_arrays_to_rgba(
            list_of_arrays_list,
            norm=norm_obj,
            chroma_boost=chroma_boost,
        )

        if figax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figax

        ax.imshow(rgba)
        ax.set(xticks=[], yticks=[])

        if cbar:
            raise NotImplementedError()

        if scalebar_config is not None:
            add_scalebar_to_ax(
                ax,
                rgba.shape[1],
                scalebar_config.sampling,
                scalebar_config.length,
                scalebar_config.units,
                scalebar_config.width_px,
                scalebar_config.pad_px,
                scalebar_config.color,
                scalebar_config.loc,
            )

        return fig, ax


    def _normalize_show_input_to_grid(
        arrays: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
    ) -> List[List[NDArray]]:
        """Convert various input formats to a consistent grid format for visualization.

        This helper function normalizes different input formats to a consistent
        grid format that can be used by the visualization functions.

        Parameters
        ----------
        arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
            Input arrays in various formats.

        Returns
        -------
        list of lists of ndarray
            Normalized grid format where each inner list represents a row of arrays.
        """
        if isinstance(arrays, np.ndarray):
            if arrays.ndim == 3:
                n_slices = arrays.shape[0]

                # Find the best divisor close to target_dim
                best_rows = 1
                best_cols = n_slices
                min_diff = abs(best_rows - best_cols)

                for i in range(1, int(np.sqrt(n_slices)) + 1):
                    if n_slices % i == 0:
                        rows, cols = i, n_slices // i
                        diff = abs(rows - cols)
                        if diff < min_diff:
                            min_diff = diff
                            best_rows, best_cols = rows, cols

                # Reshape the array into the best grid
                return [
                    arrays[i : i + best_cols].tolist()
                    for i in range(0, n_slices, best_cols)
                ]
            else:
                return [[arrays]]
        if isinstance(arrays, Sequence) and not isinstance(arrays[0], Sequence):
            # Convert sequence to list and ensure each element is an NDArray
            return [[cast(NDArray, arr) for arr in arrays]]
        # Convert outer and inner sequences to lists, ensuring proper types
        return [[cast(NDArray, arr) for arr in row] for row in arrays]

 
 
    def show_2d(
        array: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
        *,
        figax: Optional[Tuple[Any, Any]] = None,
        axsize: Tuple[int, int] = (4, 4),
        tight_layout: bool = True,
        combine_images: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Display one or more 2D arrays in a grid layout.

        This is the main visualization function that can display a single array,
        a list of arrays, or a grid of arrays. It supports both individual and
        combined visualization modes.

        Parameters
        ----------
        arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
            The arrays to visualize. Can be a single array, a sequence of arrays,
            or a nested sequence representing a grid of arrays.
        figax : tuple, optional
            (fig, axs) tuple to use for plotting. If None, a new figure and axes are created.
        axsize : tuple, default=(4, 4)
            Size of each subplot in inches.
        tight_layout : bool, default=True
            Whether to apply tight_layout to the figure.
        combine_images : bool, default=False
            If True and arrays is a sequence, combine all arrays into a single visualization
            using color encoding. Only works for a single row of arrays.
        **kwargs : dict
            Additional keyword arguments passed to _show_2d or _show_2d_combined.

        Returns
        -------
        fig : Figure
            The matplotlib figure object.
        axs : ndarray of Axes
            The matplotlib axes objects. If multiple arrays are displayed, this is a 2D array.

        Raises
        ------
        ValueError
            If combine_images is True but arrays contains multiple rows, or if
            figax is provided but the axes shape doesn't match the grid shape.
        """
        if isinstance(array, (list, tuple)):
            # Convert list of Tensors to list of numpy arrays
            return show_2d_array(
                array,
                figax=figax,
                axsize=axsize,
                tight_layout=tight_layout,
                combine_images=combine_images,
                **kwargs,
            )
        else:
            if array.ndim <= 3:
                return show_2d_array(
                    array,
                    figax=figax,
                    axsize=axsize,
                    tight_layout=tight_layout,
                    combine_images=combine_images,
                    **kwargs,
                )
            else:
                raise ValueError("array must be 2D or less")


    def show_2d_array(
        arrays: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
        *,
        figax: Optional[Tuple[Any, Any]] = None,
        axsize: Tuple[int, int] = (4, 4),
        tight_layout: bool = True,
        combine_images: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Display one or more 2D arrays in a grid layout.

        This is the main visualization function that can display a single array,
        a list of arrays, or a grid of arrays. It supports both individual and
        combined visualization modes.

        Parameters
        ----------
        arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
            The arrays to visualize. Can be a single array, a sequence of arrays,
            or a nested sequence representing a grid of arrays.
        figax : tuple, optional
            (fig, axs) tuple to use for plotting. If None, a new figure and axes are created.
        axsize : tuple, default=(4, 4)
            Size of each subplot in inches.
        tight_layout : bool, default=True
            Whether to apply tight_layout to the figure.
        combine_images : bool, default=False
            If True and arrays is a sequence, combine all arrays into a single visualization
            using color encoding. Only works for a single row of arrays.
        **kwargs : dict
            Additional keyword arguments passed to _show_2d or _show_2d_combined.

        Returns
        -------
        fig : Figure
            The matplotlib figure object.
        axs : ndarray of Axes
            The matplotlib axes objects. If multiple arrays are displayed, this is a 2D array.

        Raises
        ------
        ValueError
            If combine_images is True but arrays contains multiple rows, or if
            figax is provided but the axes shape doesn't match the grid shape.
        """
        grid = _normalize_show_input_to_grid(arrays)
        nrows = len(grid)
        ncols = max(len(row) for row in grid)

        title = kwargs.pop("title", None)

        if combine_images:
            if nrows > 1:
                raise ValueError()

            return _show_2d_combined(grid[0], figax=figax, **kwargs)

        if figax is not None:
            fig, axs = figax
            if not isinstance(axs, np.ndarray):
                axs = np.array([[axs]])
            elif axs.ndim == 1:
                axs = axs.reshape(1, -1)
            if axs.shape != (nrows, ncols):
                raise ValueError()
        else:
            fig, axs = plt.subplots(
                nrows, ncols, figsize=(axsize[0] * ncols, axsize[1] * nrows), squeeze=False
            )

        for i, row in enumerate(grid):
            for j, array in enumerate(row):
                figax = (fig, axs[i][j])
                if title is None:
                    t = None
                elif isinstance(title, str):
                    t = title
                elif isinstance(title[0], str):
                    # Flat list of titles
                    t = title[i * ncols + j] if i * ncols + j < len(title) else None
                else:
                    # Grid of titles
                    t = title[i][j] if i < len(title) and j < len(title[i]) else None

                _show_2d(
                    array,
                    figax=figax,
                    title=t,
                    **kwargs,
                )

                # figax = (fig, axs[i][j])
                # _show_2d(
                #     array,
                #     figax=figax,
                #     **kwargs,
                # )

        # Hide unused axes in incomplete rows
        for i, row in enumerate(grid):
            for j in range(len(row), ncols):
                axs[i][j].axis("off")  # type: ignore

        if tight_layout:
            fig.tight_layout()

        # Squeeze the axes to the expected shape
        if axs.shape == (1, 1):
            axs = axs[0, 0]
        elif axs.shape[0] == 1:
            axs = axs[0]
        elif axs.shape[1] == 1:
            axs = axs[:, 0]

        return fig, axs


    C10 = mo.ui.slider(start=-1000, stop=1000, step=100, label="defocus [Å]", show_value=True)
    return C10, ScalebarConfig, fftshift, ifft2, mo, np, plt, show_2d


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
def _(C10, ScalebarConfig, complex_probe, fftshift, ifft2, np, plt, show_2d):




    n = 192
    q_max = 2 # inverse Angstroms
    sampling = 1 / q_max / 2 # Angstroms
    dk = q_max / (n/2)
    wavelength = 2.5e-2 # 200kV
    q_probe = 1
    qx = qy = np.fft.fftfreq(n,sampling)
    q2 = qx[:,None]**2 + qy[None,:]**2
    q  = np.sqrt(q2)
    C30 = 0


    images = []


    reciprocal_sampling = 2 * q_max / n
    probe_fourier = complex_probe(q, wavelength, C10.value, C30, q_probe, reciprocal_sampling)
    probe_reals = fftshift(ifft2(probe_fourier))

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


@app.cell
def _(np):


    def chi(
        q,
        wavelength,
        C10,
        C30,
    ):
        """ """
        prefactor = 2*np.pi / wavelength
        alpha = q*wavelength
        order_2 = alpha**2 / 2 * C10 
        order_4 = alpha**4 / 4 * C30

        return (order_2 + order_4) * prefactor

    def complex_probe(
        q,
        wavelength,
        C10,
        C30,
        q_probe,
        reciprocal_sampling,
    ):
        """ """
        probe_array_fourier_0 = np.sqrt(
        np.clip(
                (q_probe - q)/reciprocal_sampling + 0.5,
                0,
                1,
            ),
        )
        probe_array_fourier_0 /= np.sqrt(np.sum(np.abs(probe_array_fourier_0)**2))

        chi_ = chi(
            q,
            wavelength,
            C10,
            C30,
        )

        return probe_array_fourier_0 * np.exp(-1j*chi_)


    return (complex_probe,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
