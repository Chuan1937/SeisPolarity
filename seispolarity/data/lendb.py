from .base import WaveformBenchmarkDataset


class LenDB(WaveformBenchmarkDataset):
    """
    Len-DB dataset from Magrini et al.
    Magrini 等人的 Len-DB 数据集。
    """

    def __init__(self, **kwargs):
        citation = (
            "Magrini, Fabrizio, Jozinović, Dario, Cammarano, Fabio, Michelini, Alberto, & Boschi, Lapo. "
            "(2020). LEN-DB - Local earthquakes detection: a benchmark dataset of 3-component seismograms "
            "built on a global scale [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3648232"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )
