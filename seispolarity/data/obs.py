from .base import WaveformBenchmarkDataset


class OBS(WaveformBenchmarkDataset):
    """
    OBS Benchmark Dataset of local events


    Default component order is 'Z12H'. You can easily omit one component like, e.g., hydrophone by explicitly passing
    parameter 'component_order="Z12"'. This way, the dataset can be input to land station pickers that use only 3
    components.
    默认分量顺序为 'Z12H'。您可以通过显式传递参数 'component_order="Z12"' 来轻松省略一个分量，例如水听器。这样，数据集可以输入到仅使用 3 个分量的陆地台站拾取器中。
    """

    def __init__(self, component_order="Z12H", **kwargs):
        citation = (
            "Bornstein, T., Lange, D., Münchmeyer, J., Woollam, J., Rietbrock, A., Barcheck, G., "
            "Grevemeyer, I., Tilmann, F. (2023). PickBlue: Seismic phase picking for ocean bottom "
            "seismometers with deep learning. arxiv preprint. http://arxiv.org/abs/2304.06635"
        )

        self._write_chunk_file()

        super().__init__(
            citation=citation,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [
            "201805",
            "201806",
            "201807",
            "201808",
            "201809",
            "201810",
            "201811",
            "201812",
            "201901",
            "201902",
            "201903",
            "201904",
            "201905",
            "201906",
            "201907",
            "201908",
            "000000",
        ]

    def _write_chunk_file(self):
        """
        Write out the chunk file
        写出 chunk 文件

        :return: None
        """
        chunks_path = self.path / "chunks"

        if chunks_path.is_file():
            return

        chunks = self.available_chunks()
        chunks_str = "\n".join(chunks) + "\n"

        self.path.mkdir(exist_ok=True, parents=True)
        with open(chunks_path, "w") as f:
            f.write(chunks_str)
