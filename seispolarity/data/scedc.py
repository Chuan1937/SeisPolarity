from .base import WaveformBenchmarkDataset


class SCEDC(WaveformBenchmarkDataset):
    """
    SCEDC waveform archive (2000-2020).

    Splits are set using standard random sampling of :py:class: BenchmarkDataset.
    """

    def __init__(self, **kwargs):
        citation = (
            "SCEDC (2013): Southern California Earthquake Center."
            "https://doi.org/10.7909/C3WD3xH1"
        )

        super().__init__(citation=citation, repository_lookup=True, **kwargs)


class Ross2018JGRFM(WaveformBenchmarkDataset):
    """
    First motion polarity dataset belonging to the publication:
    Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). P wave arrival picking and first‐motion polarity determination
    with deep learning. Journal of Geophysical Research: Solid Earth, 123, 5120– 5129.
    https://doi.org/10.1029/2017JB015251
    
    Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). P wave arrival picking and first‐motion polarity determination
    with deep learning. Journal of Geophysical Research: Solid Earth, 123, 5120– 5129.
    https://doi.org/10.1029/2017JB015251

    Note that this dataset contains picks as well.

    .. warning::

        This dataset only contains traces for the Z component.
        It therefore ignores the default SeisPolarity the component_order.

    """

    def __init__(self, component_order="Z", **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). "
            "P wave arrival picking and first‐motion polarity determination with deep learning. "
            "Journal of Geophysical Research: Solid Earth, 123, 5120– 5129. https://doi.org/10.1029/2017JB015251"
        )
        super().__init__(
            citation=citation,
            repository_lookup=False,
            component_order=component_order,
            **kwargs,
        )
