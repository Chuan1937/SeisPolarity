from .base import WaveformBenchmarkDataset


class ISC_EHB_DepthPhases(WaveformBenchmarkDataset):
    """
    Dataset of depth phase picks from the
    `ISC-EHB bulletin <http://www.isc.ac.uk/isc-ehb/>`_.
    来自 `ISC-EHB 公报 <http://www.isc.ac.uk/isc-ehb/>`_ 的深度相位拾取数据集。
    """

    def __init__(self, **kwargs):
        citation = (
            "Münchmeyer, J., Saul, J. & Tilmann, F. (2023) "
            "Learning the deep and the shallow: Deep learning "
            "based depth phase picking and earthquake depth estimation."
            "Seismological Research Letters (in revision)."
        )

        self._write_chunk_file()

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [str(x) for x in range(1987, 2019)]

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
