import numpy as np
from .base import WaveformBenchmarkDataset

# Conversion from earth radius
DEG2KM = 2 * np.pi * 6371 / 360


class NEIC(WaveformBenchmarkDataset):
    """
    NEIC dataset from Yeck and Patton
    """

    def __init__(self, **kwargs):
        citation = (
            "Yeck, W.L., and Patton, J., 2020, Waveform Data and Metadata used to "
            "National Earthquake Information Center Deep-Learning Models: "
            "U.S. Geological Survey data release, https://doi.org/10.5066/P9OHF4WL."
        )
        super().__init__(citation=citation, repository_lookup=True, **kwargs)


class MLAAPDE(WaveformBenchmarkDataset):
    """
    MLAAPDE dataset from Cole et al. (2023)

    Note that the SeisPolarity version is not identical to the precompiled version
    distributed directly through USGS but uses a different data selection.
    In addition, custom versions of MLAAPDE can be compiled with the software
    provided by the original authors. These datasets can be exported in
    SeisPolarity format.
    请注意，SeisPolarity 版本与直接通过 USGS 分发的预编译版本不同，而是使用了不同的数据选择。
    此外，可以使用原始作者提供的软件编译 MLAAPDE 的自定义版本。这些数据集可以导出为 SeisPolarity 格式。
    """

    def __init__(self, **kwargs):
        citation = (
            "Cole, H. M., Yeck, W. L., & Benz, H. M. (2023). "
            "MLAAPDE: A Machine Learning Dataset for Determining "
            "Global Earthquake Source Parameters. "
            "Seismological Research Letters, 94(5), 2489-2499. "
            "https://doi.org/10.1785/0220230021"
            "\n\n"
            "Cole H. M. and W. L. Yeck, 2022, "
            "Global Earthquake Machine Learning Dataset: "
            "Machine Learning Asset Aggregation of the PDE (MLAAPDE): "
            "U.S. Geological Survey data release, doi:10.5066/P96FABIB"
        )
        license = "MLAAPDE code under CC0 1.0 Universal, data licenses dependent on the underlying networks"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [
            "_201307",
            "_201308",
            "_201309",
            "_201310",
            "_201311",
            "_201312",
            "_201401",
            "_201402",
            "_201403",
            "_201404",
            "_201405",
            "_201406",
            "_201407",
            "_201408",
            "_201409",
            "_201410",
            "_201411",
            "_201412",
            "_201501",
            "_201502",
            "_201503",
            "_201504",
            "_201505",
            "_201506",
            "_201507",
            "_201508",
            "_201509",
            "_201510",
            "_201511",
            "_201512",
            "_201601",
            "_201602",
            "_201603",
            "_201604",
            "_201605",
            "_201606",
            "_201607",
            "_201608",
            "_201609",
            "_201610",
            "_201611",
            "_201612",
            "_201701",
            "_201702",
            "_201703",
            "_201704",
            "_201705",
            "_201706",
            "_201707",
            "_201708",
            "_201709",
            "_201710",
            "_201711",
            "_201712",
            "_201901",
            "_201902",
            "_201903",
            "_201904",
            "_201905",
            "_201906",
            "_201907",
            "_201908",
            "_201909",
            "_201910",
            "_201911",
            "_201912",
            "_202001",
            "_202002",
            "_202003",
            "_202004",
            "_202005",
            "_202006",
            "_202007",
            "_202008",
            "_202009",
            "_202010",
            "_202011",
            "_202012",
            "_202101",
            "_202102",
            "_202103",
            "_202104",
            "_202105",
            "_202106",
            "_202107",
            "_202108",
            "_202109",
            "_202110",
            "_202111",
            "_202112",
            "_201801",
            "_201802",
            "_201803",
            "_201804",
            "_201805",
            "_201806",
            "_201807",
            "_201808",
            "_201809",
            "_201811",
            "_201810",
            "_201812",
            "_202201",
            "_202202",
            "_202203",
            "_202204",
        ]
