from .base import WaveformBenchmarkDataset


class CWABase(WaveformBenchmarkDataset):
    """
    Base class for CWA datasets.
    CWA 数据集的基类。
    """

    def __init__(self, **kwargs):
        citation = (
            "Chien-Ying Wang, Hao-Ting Huang, Jui-Chi Hung, "
            "Chien-Chih Chen, Strong-Motion Records of the 1999 Chi-Chi, "
            "Taiwan, Earthquake, Seismological Research Letters, "
            "Volume 73, Issue 2, March/April 2002, Pages 199–207, "
            "https://doi.org/10.1785/gssrl.73.2.199"
        )

        super().__init__(
            citation=citation,
            repository_lookup=True,
            **kwargs,
        )


class CWA(CWABase):
    """
    CWA dataset - Events and traces.
    CWA 数据集 - 事件和波形。
    """

    pass


class CWANoise(CWABase):
    """
    CWA dataset - Noise samples.
    CWA 数据集 - 噪声样本。
    """

    pass
