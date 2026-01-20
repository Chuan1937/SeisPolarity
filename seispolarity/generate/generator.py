from torch.utils.data import Dataset
import numpy as np


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

    @property
    def augmentations(self):
        """获取增强函数列表（与WaveformDataset兼容）"""
        return self._augmentations

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

        # Check if sample is already a state_dict (dict)
        if isinstance(sample, dict) and "X" in sample:
            return sample

        return {"X": sample}

    def _clean_state_dict(self, state_dict):
        cleaned_state_dict = {}

        for k, v in state_dict.items():
            if isinstance(v, tuple) and len(v) == 2:
                metadata = v[1]
                if isinstance(metadata, dict) or metadata is None:
                    # 对于X键，保持为(data, metadata)元组格式
                    # 因为MetadataToLabel需要metadata来提取标签
                    cleaned_state_dict[k] = v
                else:
                    # 如果metadata不是dict或None，但数据是有效的，我们仍然保留它
                    # 这可能发生在某些增强函数修改了数据结构的情况下
                    cleaned_state_dict[k] = v
            else:
                # 如果值不是元组，我们仍然保留它
                # 这可能发生在某些增强函数修改了数据结构的情况下
                cleaned_state_dict[k] = v

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


class BalancedPolarityGenerator(GenericGenerator):
    """
    平衡极性数据生成器，用于处理类别不平衡问题。
    
    这个生成器专门用于极性分类任务，提供以下功能：
    1. 对U（正向）和D（负向）类别进行1:1平衡
    2. 从X（不确定）类别中采样与U+D相同数量的样本
    3. 如果X数量不够，会自动复制X样本
    4. 可选地应用极性反转增强
    
    使用方式：
    ```python
    # 创建平衡生成器
    balanced_generator = BalancedPolarityGenerator(
        dataset=waveform_dataset,
        apply_polarity_inversion=True  # 可选，默认为True
    )
    
    # 添加其他增强
    balanced_generator.add_augmentations([
        Demean(key="X", axis=-1),
        Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    ])
    ```
    """
    def __init__(self, dataset, apply_polarity_inversion=True, label_key="label", 
                 label_map={'U': 0, 'D': 1, 'X': 2}, random_seed=42, 
                 inherit_augmentations=False):
        super().__init__(dataset)
        self.label_key = label_key
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 可选择是否继承dataset的增强
        if inherit_augmentations and hasattr(dataset, 'augmentations') and dataset.augmentations:
            print(f"继承数据集中的 {len(dataset.augmentations)} 个增强")
            self.add_augmentations(dataset.augmentations)
        
        # 构建平衡索引
        self._build_balanced_indices()
        
        # 如果需要，添加极性反转增强
        if apply_polarity_inversion:
            from .augmentation import PolarityInversion
            self.add_augmentations([
                PolarityInversion(key="X", label_key=label_key, label_map=label_map)
            ])
    
    def _build_balanced_indices(self):
        """构建平衡的样本索引，处理X类别不足的情况"""
        # 获取所有样本的标签
        labels = []
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                # 处理不同的返回格式
                if isinstance(sample, tuple) and len(sample) == 2:
                    _, metadata = sample
                elif isinstance(sample, dict) and "X" in sample:
                    # 处理GenericGenerator返回的state_dict格式
                    x_value = sample["X"]
                    if isinstance(x_value, tuple) and len(x_value) == 2:
                        _, metadata = x_value
                    else:
                        metadata = {}
                else:
                    # 未知格式，跳过
                    print(f"警告: 样本{i}的返回格式无法识别: {type(sample)}")
                    labels.append(-1)
                    continue
                
                raw_label = metadata.get(self.label_key, -1)
                
                # 处理字节字符串标签
                if isinstance(raw_label, bytes):
                    label_str = raw_label.decode('utf-8').upper()
                elif isinstance(raw_label, str):
                    label_str = raw_label.upper()
                elif isinstance(raw_label, (int, np.integer)):
                    # 如果是数字标签，直接使用
                    labels.append(raw_label)
                    continue
                else:
                    # 无法识别的标签类型
                    print(f"警告: 样本{i}的标签类型无法识别: {type(raw_label)}")
                    labels.append(-1)
                    continue
                
                # 将字符串标签映射为数字
                label_num = self.label_map.get(label_str, -1)
                labels.append(label_num)
                
            except Exception as e:
                print(f"警告: 无法获取样本{i}的标签: {e}")
                labels.append(-1)  # 无效标签
        
        labels = np.array(labels)
        
        # 分离不同类别的索引
        u_indices = np.where(labels == self.label_map['U'])[0]
        d_indices = np.where(labels == self.label_map['D'])[0]
        x_indices = np.where(labels == self.label_map['X'])[0]
        
        print(f"原始数据分布: U={len(u_indices)}, D={len(d_indices)}, X={len(x_indices)}")
        
        # 确定最小类别数量
        min_ud = min(len(u_indices), len(d_indices))
        
        if min_ud == 0:
            raise ValueError("U或D类别没有样本，无法进行平衡")
        
        # 从U和D类别中随机选择相同数量的样本
        u_selected = np.random.choice(u_indices, size=min_ud, replace=False)
        d_selected = np.random.choice(d_indices, size=min_ud, replace=False)
        
        # 计算需要的X样本数量（与U，D相同）
        needed_x_count =min_ud
        
        # 处理X类别样本不足的情况
        if len(x_indices) == 0:
            print("警告: X类别没有样本，将使用U和D样本")
            x_selected = np.array([], dtype=np.int64)
        elif len(x_indices) < needed_x_count:
            print(f"警告: X类别样本不足 ({len(x_indices)} < {needed_x_count})，将复制X样本")
            # 复制X样本直到达到所需数量
            repeats = needed_x_count // len(x_indices) + 1
            x_repeated = np.tile(x_indices, repeats)[:needed_x_count]
            x_selected = x_repeated
        else:
            # 从X类别中选择与U+D相同数量的样本
            x_selected = np.random.choice(x_indices, size=needed_x_count, replace=False)
        
        # 合并所有选择的索引
        self.balanced_indices = np.concatenate([u_selected, d_selected, x_selected])
        np.random.shuffle(self.balanced_indices)  # 打乱顺序
        
        # 计算最终分布
        final_labels = labels[self.balanced_indices]
        final_u = np.sum(final_labels == self.label_map['U'])
        final_d = np.sum(final_labels == self.label_map['D'])
        final_x = np.sum(final_labels == self.label_map['X'])
        
        print(f"平衡后数据分布: U={final_u}, D={final_d}, X={final_x}, 总计={len(self.balanced_indices)}")
    
    def __len__(self):
        return len(self.balanced_indices)
    
    def _populate_state_dict(self, idx):
        # 使用平衡索引获取样本
        original_idx = self.balanced_indices[idx]
        
        # 从原始数据集中获取样本
        if hasattr(self.dataset, "get_sample"):
            data, metadata = self.dataset.get_sample(original_idx)
        else:
            data, metadata = self.dataset[original_idx]
        
        # 确保元数据中包含标签信息（MetadataToLabel需要）
        # 如果元数据中没有标签键，添加一个
        if self.label_key not in metadata:
            # 尝试从原始数据中获取标签
            raw_label = metadata.get(self.label_key, -1)
            metadata[self.label_key] = raw_label
        
        # 创建包含元数据的state_dict
        # 注意：X必须是(data, metadata)元组格式，因为MetadataToLabel需要metadata
        state_dict = {
            "X": (data, metadata)
            # 不再单独存储label键，因为MetadataToLabel会从X的元数据中提取
        }
        
        return state_dict
