from .base import WaveformBenchmarkDataset


class GEOFON(WaveformBenchmarkDataset):
    """
    GEOFON dataset consisting of both regional and teleseismic picks. Mostly contains P arrivals,
    but a few S arrivals are annotated as well. Contains data from 2010-2013. The dataset will be
    downloaded from the SeisPolarity repository on first usage.
    GEOFON 数据集包含区域和远震拾取。主要包含 P 波到达，但也标注了一些 S 波到达。包含 2010-2013 年的数据。
    """

    def __init__(self, **kwargs):
        # TODO: Add citation
        citation = "GEOFON dataset"
        super().__init__(citation=citation, repository_lookup=True, **kwargs)
