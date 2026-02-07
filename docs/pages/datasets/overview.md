# Datasets Overview

SeisPolarity provides a unified interface to access and process multiple seismic polarity datasets. All datasets are stored in HDF5 format for efficient storage and streaming.

## Available Datasets

| Dataset | Source | Size | Classes | Format |
|---------|--------|------|---------|--------|
| SCSN | Southern California Seismic Network | Large | U/D/N | HDF5 |
| Txed | Texas Earthquake Dataset | Medium | U/D | HDF5 |
| DiTing | China Earthquake Networks Center | Medium | U/D/N | HDF5 |
| Instance | Global seismic data | Medium | U/D/N | HDF5 |
| PNW | Pacific Northwest | Large | U/D/N | HDF5 |

## Datasets

### SCSN

The Southern California Seismic Network (SCSN) dataset contains polarity-labeled seismic waveforms from the Southern California Seismic Network. This dataset covers earthquake events from 2000-2020 and is a high-quality dataset with manual annotations.

```{image} ../datasets/SCSN.png
:alt: SCSN
:align: center
```

#### Warning

Dataset size: waveforms.hdf5 ~660Gb, metadata.csv ~2.2Gb
Polarity subset SCSN ~15 GB
 
#### Citation

Cheng, Y., Ross, Z. E., Hauksson, E., Ben-Zion, Y. (2023). Refined earthquake focal mechanism catalog for southern California derived with deep learning algorithms. Journal of Geophysical Research: Solid Earth, 128, e2022JB025975. https://doi.org/10.1029/2022JB025975

Ross, Z. E., Meier, M.-A., Hauksson, E. (2018). P wave arrival picking and first-motion polarity determination with deep learning. Journal of Geophysical Research: Solid Earth, 123, 5405-5416. https://doi.org/10.1029/2018JB015510

---

### Txed

The Texas Earthquake Dataset (TXED) is a regional seismic signal benchmark dataset from Texas. This dataset contains a large number of earthquake events and noise waveforms, serving as an important data resource for machine learning in seismology.

```{image} ../datasets/TXED.png
:alt: Txed
:align: center
```

#### Warning

Dataset size: waveforms.hdf5 ~70Gb, metadata.csv 120Mb

#### Citation

Chen, Y., Savvaidis, A., Saad, O. M., Huang, G.-C. D., Siervo, D., O'Sullivan, V., McCabe, C., Uku, B., Fleck, P., Burke, G., Alvarez, N. L., Domino, J., & Grigoratos, I. (2024). TXED: The Texas Earthquake Dataset for AI. Seismological Research Letters, 95(6), 1-13. https://doi.org/10.1785/0220230327

---

### DiTing

The DiTing dataset is a large-scale Chinese seismic benchmark dataset specifically designed for artificial intelligence seismology research. This dataset contains over 640,000 high-quality P-wave first-motion polarity labels, covering more than 1,300 broadband and short-period seismic stations across China.

```{image} ../datasets/DITING.jpg
:alt: Diting
:align: center
```

#### Citation

Zhao, M., Xiao, Z., Chen, S., & Fang, L. (2023). DiTing: A large-scale Chinese seismic benchmark dataset for artificial intelligence in seismology. Earthquake Science, 36(2), 84-94. https://doi.org/10.1016/j.eqs.2022.01.022

#### Data Download

The dataset can be requested for download at: https://data.earthquake.cn/

---

### Instance

The INSTANCE dataset is an Italian seismic waveform dataset compiled by the Italian National Institute of Geophysics and Volcanology (INGV), specifically designed for machine learning applications. This dataset contains nearly 1.2 million three-component waveform traces and serves as an important resource for seismological research.

```{image} ../datasets/INSTANCE.png
:alt: Instance
:align: center
```

#### Warning

Dataset size:
- waveforms (counts) ~160Gb
- waveforms (ground motion units) ~310Gb

#### Citation

Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). INSTANCE – The Italian Seismic Dataset For Machine Learning. Earth System Science Data, 13, 5509–5542. https://doi.org/10.5194/essd-13-5509-2021

---

### PNW

The Pacific Northwest (PNW) dataset is a machine learning-ready curated dataset containing diverse seismic signals from the Pacific Northwest region. This dataset is compiled by the Pacific Northwest Seismic Network and covers various seismic event types including earthquakes, explosions, and noise.

```{image} ../datasets/PNW.png
:alt: PNW
:align: center
```

#### Citation

Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., & Wright, A. (2023). Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1), 368. https://doi.org/10.26443/seismica.v2i1.368

## Loading Datasets

### Automatic Download

SeisPolarity can automatically download datasets:

```python
from seispolarity import get_dataset_path, WaveformDataset

# Download from Hugging Face (default)
data_path = get_dataset_path("SCSN", "train", cache_dir="./datasets")

# Or use ModelScope (recommended for users in China)
data_path = get_dataset_path("SCSN", "train", use_hf=False)
```

### Load from Local Files

```python
from seispolarity import WaveformDataset

# Disk streaming (suitable for large datasets)
dataset = WaveformDataset(
    path="data/scsn_train.hdf5",
    name="SCSN_Train",
    preload=False
)

# RAM preloading (suitable for small datasets)
dataset = WaveformDataset(
    path="data/scsn_train.hdf5",
    name="SCSN_Train",
    preload=True
)
```

## Dataset API

### WaveformDataset

The main class for loading waveform data.

```python
from seispolarity import WaveformDataset

dataset = WaveformDataset(
    path="data.hdf5",          # HDF5 file path
    name="SCSN",               # Dataset name
    preload=False,             # Whether to preload into RAM
    data_key="X",              # HDF5 key for waveforms
    label_key="Y",             # HDF5 key for labels
    p_pick_position=300,      # P-wave arrival position
    pick_key="p_pick",        # Use p_pick as P-wave arrival point
    crop_left=200,             # Samples before P-pick
    crop_right=200,            # Samples after P-pick
    allowed_labels=[0, 1, 2]   # Allowed labels (0: Up, 1: Down, 2: Unknown)
)
```

### Data Format

Waveforms are stored in HDF5 files with the following structure:

```
waveforms.hdf5
├── X                # Waveform data (N_samples, N_channels)
├── Y                # P-value labels (N_samples,)
├── Z                # Clarity (only required for ditingmotion)   
├── metadata         # Additional metadata (optional)
└── ...
```

### Label Encoding

- **0**: Up (positive polarity)
- **1**: Down (negative polarity)
- **2**: Unknown

### DataLoader

Create a PyTorch DataLoader for training:

```python
loader = dataset.get_dataloader(
    batch_size=1024,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)
```

## Data Inspection

### Basic Statistics

```python
from seispolarity import WaveformDataset

dataset = WaveformDataset(path="data.hdf5", name="SCSN")

# Get dataset statistics
print(f"Total samples: {len(dataset)}")
print(f"Label distribution: {dataset.label_distribution}")
print(f"Waveform shape: {dataset.waveform_shape}")
```

## Multi-Dataset Training

Combine multiple datasets:

```python
from seispolarity import MultiWaveformDataset

# Create multiple datasets
dataset1 = WaveformDataset(path="scsn.hdf5", name="SCSN")
dataset2 = WaveformDataset(path="txed.hdf5", name="Txed")

# Combine them
combined = MultiWaveformDataset([dataset1, dataset2])
```

## Balanced Sampling

For datasets with label imbalance, use balanced sampling:

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"  # or "min_based"
)
loader = generator.get_dataloader(batch_size=256)
```

## Download Locations

Datasets can be downloaded from:

- **Hugging Face**: `https://huggingface.co/datasets/chuanjun1978/Seismic-AI-Data`
- **ModelScope**: `https://www.modelscope.cn/datasets/chuanjun/Seismic-AI-Data/` (recommended for users in China)

For more details, see the [Installation Guide](../installation.md).
