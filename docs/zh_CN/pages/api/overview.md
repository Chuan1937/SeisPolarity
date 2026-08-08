# API 参考

本节提供 SeisPolarity 各模块的详细 API 文档。

## 模块

- [数据集 API](dataset.md) - WaveformDataset 与数据加载工具
- [模型 API](model.md) - 预训练模型与模型工具
- [训练 API](training.md) - 训练工具与配置
- [增强 API](augmentation.md) - 数据增强类

## 包结构

```
seispolarity/
├── __init__.py              # 主包入口
├── dataset/                 # 数据集加载与工具
│   ├── __init__.py
│   ├── dataset.py          # WaveformDataset 类
│   ├── utils.py            # 数据集工具
│   └── huggingface.py      # Hugging Face 集成
├── models/                  # 模型结构
│   ├── __init__.py
│   ├── ross.py             # Ross 模型
│   ├── eqpolarity.py       # Eqpolarity 模型
│   ├── ditingmotion.py     # DiTingMotion 模型
│   ├── cfm.py              # CFM 模型
│   ├── rpnet.py            # RPNet 模型
│   ├── polarcap.py         # PolarCAP 模型
│   └── app.py              # APP 模型
├── training/                # 训练工具
│   ├── __init__.py
│   ├── trainer.py          # Trainer 类
│   └── config.py           # TrainingConfig 类
└── generate/                # 数据生成与增强
    ├── __init__.py
    ├── generator.py        # 生成器类
    └── augmentation.py     # 增强类
```

## 快速链接

### 数据加载

```{eval-rst}
.. autoclass:: seispolarity.dataset.WaveformDataset
   :members:
```

### 模型

```{eval-rst}
.. autoclass:: seispolarity.models.PPNet
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.models.EqpolarityNet
   :members:
```

### 训练

```{eval-rst}
.. autoclass:: seispolarity.training.Trainer
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.training.TrainingConfig
   :members:
```

### 增强

```{eval-rst}
.. autoclass:: seispolarity.generate.GenericGenerator
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.generate.BalancedPolarityGenerator
   :members:
```

## 工具函数

### get_dataset_path

```{eval-rst}
.. autofunction:: seispolarity.get_dataset_path
```

### configure_cache

```{eval-rst}
.. autofunction:: seispolarity.configure_cache
```

### get_hf_path

```{eval-rst}
.. autofunction:: seispolarity.get_hf_path
```

## 异常

```{eval-rst}
.. autoclass:: seispolarity.dataset.DatasetError
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.models.ModelError
   :members:
```

## 版本信息

```python
import seispolarity

print(seispolarity.__version__)
```
