from .base import WaveformBenchmarkDataset


class CREW(WaveformBenchmarkDataset):
    """
    Curated Regional Earthquake Waveforms (CREW dataset)
    """

    def __init__(self, **kwargs):
        citation = (
            "Aguilar Suarez, A. L., & Beroza, G. (2024). "
            "Curated Regional Earthquake Waveforms (CREW) Dataset. "
            "Seismica, 3(1). https://doi.org/10.26443/seismica.v3i1.1049"
        )

        self._write_chunk_file()

        super().__init__(
            citation=citation,
            repository_lookup=True,
            **kwargs,
        )

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [
            "000",
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
            "011",
            "012",
            "013",
            "014",
            "015",
            "016",
            "017",
            "018",
            "019",
            "020",
            "021",
            "022",
            "023",
            "024",
            "025",
            "026",
            "027",
            "028",
            "029",
            "030",
            "031",
            "032",
            "033",
            "034",
            "035",
            "036",
            "037",
            "038",
            "039",
            "040",
            "041",
            "042",
            "043",
            "044",
            "045",
            "046",
            "047",
            "048",
            "049",
            "050",
            "052",
            "053",
            "054",
            "055",
            "056",
            "057",
            "058",
            "059",
            "060",
            "061",
            "062",
            "063",
            "064",
            "065",
            "066",
            "067",
            "068",
            "069",
            "070",
            "071",
            "072",
            "073",
        ]
