from .base import WaveformBenchmarkDataset


class LFEStacksCascadiaBostock2015(WaveformBenchmarkDataset):
    """
    Low-frequency earthquake stacks underneath Vancouver Island, Cascadia, Canada/USA based on the catalog by
    Bostock et al (2015). Compiled to SeisPolarity format by Münchmeyer et al (2024).
    基于 Bostock 等人 (2015) 目录的加拿大/美国卡斯卡迪亚温哥华岛下方的低频地震叠加。由 Münchmeyer 等人 (2024) 编译为 SeisPolarity 格式。
    """

    def __init__(self, component_order="Z12", **kwargs):
        citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )


class LFEStacksMexicoFrank2014(WaveformBenchmarkDataset):
    """
    Low-frequency earthquake stacks underneath Guerrero, Mexico based on the catalog by
    Frank et al (2014). Compiled to SeisPolarity format by Münchmeyer et al (2024).
    基于 Frank 等人 (2014) 目录的墨西哥格雷罗下方的低频地震叠加。由 Münchmeyer 等人 (2024) 编译为 SeisPolarity 格式。
    """

    def __init__(self, component_order="Z12", **kwargs):
        citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )


class LFEStacksSanAndreasShelly2017(WaveformBenchmarkDataset):
    """
    Low-frequency earthquake stacks on the San Andreas Fault, California, USA based on the catalog by
    Shelly (2014). Compiled to SeisPolarity format by Münchmeyer et al (2024).
    基于 Shelly (2014) 目录的美国加利福尼亚州圣安德烈亚斯断层上的低频地震叠加。由 Münchmeyer 等人 (2024) 编译为 SeisPolarity 格式。
    """

    def __init__(self, component_order="Z12", **kwargs):
        citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )
