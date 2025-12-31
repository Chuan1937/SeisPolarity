from .base import WaveformBenchmarkDataset


class VCSEIS(WaveformBenchmarkDataset):
    """
    A data set of seismic waveforms from various volcanic regions: Alaska, Hawaii, Northern California, Cascade volcanoes.
    来自不同火山地区的地震波形数据集：阿拉斯加、夏威夷、北加州、喀斯喀特火山。
    """

    def __init__(self, **kwargs):
        citation = (
            "Zhong, Y., & Tan, Y. J. (2024). Deep-learning-based phase "
            "picking for volcano-tectonic and long-period earthquakes. "
            "Geophysical Research Letters, 51, e2024GL108438. "
            "https://doi.org/10.1029/2024GL108438"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            **kwargs,
        )

    def get_long_period_earthquakes(self):
        """
        Return the subset with only long-period earthquakes
        返回仅包含长周期地震的子集
        """
        return self.filter(
            self.metadata["source_type"] == "lp",
            inplace=False,
        )

    def get_regular_earthquakes(self):
        """
        Return the subset with only regular earthquakes
        返回仅包含常规地震的子集
        """
        return self.filter(
            (
                (self.metadata["source_type"] != "lp")
                & (self.metadata["source_type"] != "noise")
            ),
            inplace=False,
        )

    def get_noise_traces(self):
        """
        Return the subset with only noise traces
        返回仅包含噪声轨迹的子集
        """
        return self.filter(self.metadata["source_type"] == "noise", inplace=False)

    def get_alaska_subset(self):
        """
        Select and return the data from Alaska
        选择并返回来自阿拉斯加的数据
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(["_ak_lp", "_ak_rg", "_aknoise"]),
            inplace=False,
        )

    def get_hawaii_subset(self):
        """
        Select and return the data from Hawaii
        选择并返回来自夏威夷的数据
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(
                ["_hw12t21_lp", "_hw12t21_rg", "_hwnoise"]
            ),
            inplace=False,
        )

    def get_northern_california_subset(self):
        """
        Select and return the data from Northern California
        选择并返回来自北加州的数据
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(["_ncedc_lp", "_ncedc_vt"]), inplace=False
        )

    def get_cascade_subset(self):
        """
        Select and return the data from Cascade volcanoes
        选择并返回来自喀斯喀特火山的数据
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(["_cascade_lp", "_cascade_vt"]),
            inplace=False,
        )
