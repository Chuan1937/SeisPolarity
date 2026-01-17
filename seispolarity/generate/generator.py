from torch.utils.data import Dataset


class GenericGenerator(Dataset):
    """
    通用数据生成器，用于构建数据预处理和增强流水线。
    继承自PyTorch的Dataset类，可以直接与DataLoader一起使用。
    
    生成器的处理流水线通过一系列处理步骤或增强函数定义。
    对于每个数据样本，生成器按顺序调用增强函数。
    增强函数之间的信息通过状态字典（state_dict）传递。
    
    状态字典是一个Python字典，将键映射到元组（数据, 元数据）。
    在__getitem__中，生成器自动使用键"X"将波形数据和对应的元数据填充到初始字典中。
    应用所有增强后，生成器会移除所有元数据信息。
    
    这意味着输出字典只将键映射到数据部分。
    任何应该输出的元数据都需要显式地写入到数据中。
    
    :param dataset: 底层SeisPolarity数据集
    :type dataset: seispolarity.data.WaveformDataset 或 seispolarity.data.MultiWaveformDataset
    """

    def __init__(self, dataset):
        self._augmentations = []
        self.dataset = dataset
        super().__init__()

    def augmentation(self, f):
        """
        Decorator for augmentations.
        """
        self._augmentations.append(f)

        return f

    def add_augmentations(self, augmentations):
        """
        向生成器添加增强函数列表。不能用作装饰器。
        
        :param augmentations: List of augmentations
        :type augmentations: list[callable]
        """
        if not isinstance(augmentations, list):
            raise TypeError(
                "The argument of add_augmentations must be a list of augmentations."
            )

        self._augmentations.extend(augmentations)

    def __str__(self):
        summary = f"{self.__class__} with {len(self._augmentations)} augmentations:\n"
        for i, aug in enumerate(self._augmentations):
            summary += f" {i + 1}.\t{str(aug)}\n"
        return summary

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        state_dict = self._populate_state_dict(idx)

        # Recursive application of augmentation processing methods
        for func in self._augmentations:
            func(state_dict)

        state_dict = self._clean_state_dict(state_dict)

        return state_dict

    def _populate_state_dict(self, idx):
        if hasattr(self.dataset, "get_sample"):
            sample = self.dataset.get_sample(idx)
        else:
            # Fallback: assume __getitem__ returns (data, metadata)
            sample = self.dataset[idx]
        return {"X": sample}

    def _clean_state_dict(self, state_dict):
        cleaned_state_dict = {}

        for k, v in state_dict.items():
            if isinstance(v, tuple) and len(v) == 2:
                metadata = v[1]
                if isinstance(metadata, dict) or metadata is None:
                    # Remove all metadata from the output
                    cleaned_state_dict[k] = v[0]
                else:
                    raise ValueError(f"Metadata for key '{k}' is not a dict or None.")
            else:
                raise ValueError(
                    f"Value for key '{k}' does not follow the scheme (data, metadata)."
                )

        return cleaned_state_dict


class SteeredGenerator(GenericGenerator):
    """
    受控数据生成器，继承自GenericGenerator。
    与GenericGenerator不同，此生成器由包含控制信息的数据框控制。
    控制数据框中的每一行对应生成器输出的一个样本。
    
    数据框包含两种信息：
    1. 识别迹线的信息（trace_name必需，trace_chunk和trace_dataset可选）
    2. 增强的附加信息，存储在state_dict["_control_"]中作为字典
    
    此生成器特别适用于评估，例如从迹线中提取预定义窗口时。
    
    :param dataset: 底层SeisPolarity数据集
    :type dataset: seispolarity.data.WaveformDataset 或 seispolarity.data.MultiWaveformDataset
    :param metadata: 附加信息，作为pandas数据框
    :type metadata: pandas.DataFrame
    """

    def __init__(self, dataset, metadata):
        self.metadata = metadata
        super().__init__(dataset)

    def __len__(self):
        return len(self.metadata)

    def _populate_state_dict(self, idx):
        control = self.metadata.iloc[idx].to_dict()
        kwargs = {
            "trace_name": control["trace_name"],
            "chunk": control.get("trace_chunk", None),
            "dataset": control.get("trace_dataset", None),
        }
        data_idx = self.dataset.get_idx_from_trace_name(**kwargs)

        return {"X": self.dataset.get_sample(data_idx), "_control_": control}

    def _clean_state_dict(self, state_dict):
        # Remove control information
        del state_dict["_control_"]
        return super()._clean_state_dict(state_dict)


class GroupGenerator(GenericGenerator):
    """
    组数据生成器，继承自GenericGenerator。
    与GenericGenerator不同，此生成器总是将组加载到状态字典中，而不是单个迹线。
    底层数据集的`grouping`参数需要被设置。
    """

    def __init__(self, dataset):
        if dataset.grouping is None:
            raise ValueError("Grouping needs to be set in dataset.")

        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset.groups)

    def _populate_state_dict(self, idx):
        return {"X": self.dataset.get_group_samples(idx)}
