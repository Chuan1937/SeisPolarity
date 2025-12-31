from abc import ABC
from .base import WaveformBenchmarkDataset, MultiWaveformDataset


class InstanceTypeDataset(WaveformBenchmarkDataset, ABC):
    """
    Abstract class for all datasets in the INSTANCE structure.
    """
    pass


class InstanceNoise(InstanceTypeDataset):
    """
    INSTANCE dataset - Noise samples
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). "
            "INSTANCE - The Italian Seismic Dataset For Machine Learning. "
            "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/INSTANCE"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )


class InstanceCounts(InstanceTypeDataset):
    """
    INSTANCE dataset - Events with waveforms in counts
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). "
            "INSTANCE - The Italian Seismic Dataset For Machine Learning. "
            "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/INSTANCE"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )


class InstanceGM(InstanceTypeDataset):
    """
    INSTANCE dataset - Events with waveforms in ground motion units
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). "
            "INSTANCE - The Italian Seismic Dataset For Machine Learning. "
            "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/INSTANCE"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )


class InstanceCountsCombined(MultiWaveformDataset):
    """
    Convenience class to jointly load :py:class:`InstanceCounts` and :py:class:`InstanceNoise`.

    :param kwargs: Passed to the constructors of both :py:class:`InstanceCounts` and :py:class:`InstanceNoise`
    """

    def __init__(self, **kwargs):
        super().__init__([InstanceCounts(**kwargs), InstanceNoise(**kwargs)])
