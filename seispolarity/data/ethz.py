from .base import WaveformBenchmarkDataset


class ETHZ(WaveformBenchmarkDataset):
    """
    ETHZ dataset.
    ETHZ 数据集。
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S. et al. "
            "INSTANCE – the Italian seismic dataset for machine learning. "
            "Sci Data 8, 250 (2021). https://doi.org/10.1038/s41597-021-01027-z"
        )

        super().__init__(
            citation=citation,
            repository_lookup=True,
            **kwargs,
        )
