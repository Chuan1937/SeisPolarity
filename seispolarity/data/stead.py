from .base import WaveformBenchmarkDataset


class STEAD(WaveformBenchmarkDataset):
    """
    STEAD dataset from Mousavi et al.
 
    Using the train/test split from the EQTransformer Github repository
    train/dev split defined in SeisPolarity
    """

    def __init__(self, **kwargs):
        citation = (
            "Mousavi, S. M., Sheng, Y., Zhu, W., Beroza G.C., (2019). STanford EArthquake Dataset (STEAD): "
            "A Global Data Set of Seismic Signals for AI, IEEE Access, doi:10.1109/ACCESS.2019.2947848"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )
