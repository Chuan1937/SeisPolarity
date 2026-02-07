from torch.utils.data import Dataset
import numpy as np


class GenericGenerator(Dataset):
    """
    ，。
    PyTorchDataset，DataLoader。

    。
    ，。
    （state_dict）。

    Python，（, ）。
    __getitem__，"X"。
    ，。

    。
    。

    :param dataset: SeisPolarity
    :type dataset: seispolarity.data.WaveformDataset  seispolarity.data.MultiWaveformDataset
    """

    def __init__(self, dataset):
        self._augmentations = []
        self.dataset = dataset
        super().__init__()

    @property
    def augmentations(self):
        """（WaveformDataset）"""
        return self._augmentations

    def augmentation(self, f):
        """
        Decorator for augmentations.
        """
        self._augmentations.append(f)

        return f

    def add_augmentations(self, augmentations):
        """
        。。

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
                    # X，(data, metadata)
                    # MetadataToLabelmetadata
                    cleaned_state_dict[k] = v
                else:
                    # metadatadictNone，，
                    # 
                    cleaned_state_dict[k] = v
            else:
                # ，
                # 
                cleaned_state_dict[k] = v

        return cleaned_state_dict


class SteeredGenerator(GenericGenerator):
    """
    ，GenericGenerator。
    GenericGenerator，。
    。

    ：
    1. （trace_name，trace_chunktrace_dataset）
    2. ，state_dict["_control_"]

    ，。

    :param dataset: SeisPolarity
    :type dataset: seispolarity.data.WaveformDataset  seispolarity.data.MultiWaveformDataset
    :param metadata: ，pandas
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
    ，GenericGenerator。
    GenericGenerator，，。
    `grouping`。
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
    ，。

    ，：
    1. U（）D（）1:1
    2. X（）U+D
    3. X，X
    4.

    ：
    ```python
    #
    balanced_generator = BalancedPolarityGenerator(
        dataset=waveform_dataset,
        apply_polarity_inversion=True  # ，True
    )

    #
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
        
        # dataset
        if inherit_augmentations and hasattr(dataset, 'augmentations') and dataset.augmentations:
            print(f" {len(dataset.augmentations)} ")
            self.add_augmentations(dataset.augmentations)
        
        # 
        self._build_balanced_indices()
        
        # ，
        if apply_polarity_inversion:
            from .augmentation import PolarityInversion
            self.add_augmentations([
                PolarityInversion(key="X", label_key=label_key, label_map=label_map)
            ])
    
    def _build_balanced_indices(self):
        """，X"""
        # 
        labels = []
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                # 
                if isinstance(sample, tuple) and len(sample) == 2:
                    _, metadata = sample
                elif isinstance(sample, dict) and "X" in sample:
                    # GenericGeneratorstate_dict
                    x_value = sample["X"]
                    if isinstance(x_value, tuple) and len(x_value) == 2:
                        _, metadata = x_value
                    else:
                        metadata = {}
                else:
                    # ，
                    print(f": {i}: {type(sample)}")
                    labels.append(-1)
                    continue
                
                raw_label = metadata.get(self.label_key, -1)
                
                # 
                if isinstance(raw_label, bytes):
                    label_str = raw_label.decode('utf-8').upper()
                elif isinstance(raw_label, str):
                    label_str = raw_label.upper()
                elif isinstance(raw_label, (int, np.integer)):
                    # ，
                    labels.append(raw_label)
                    continue
                else:
                    # 
                    print(f": {i}: {type(raw_label)}")
                    labels.append(-1)
                    continue
                
                # 
                label_num = self.label_map.get(label_str, -1)
                labels.append(label_num)
                
            except Exception as e:
                print(f": {i}: {e}")
                labels.append(-1)  # 
        
        labels = np.array(labels)
        
        # 
        u_indices = np.where(labels == self.label_map['U'])[0]
        d_indices = np.where(labels == self.label_map['D'])[0]
        x_indices = np.where(labels == self.label_map['X'])[0]
        
        print(f": U={len(u_indices)}, D={len(d_indices)}, X={len(x_indices)}")
        
        # 
        min_ud = min(len(u_indices), len(d_indices))
        
        if min_ud == 0:
            raise ValueError("UD，")
        
        # UD
        u_selected = np.random.choice(u_indices, size=min_ud, replace=False)
        d_selected = np.random.choice(d_indices, size=min_ud, replace=False)
        
        # X（U，D）
        needed_x_count =min_ud
        
        # X
        if len(x_indices) == 0:
            print(": X，UD")
            x_selected = np.array([], dtype=np.int64)
        elif len(x_indices) < needed_x_count:
            print(f": X ({len(x_indices)} < {needed_x_count})，X")
            # X
            repeats = needed_x_count // len(x_indices) + 1
            x_repeated = np.tile(x_indices, repeats)[:needed_x_count]
            x_selected = x_repeated
        else:
            # XU+D
            x_selected = np.random.choice(x_indices, size=needed_x_count, replace=False)
        
        # 
        self.balanced_indices = np.concatenate([u_selected, d_selected, x_selected])
        np.random.shuffle(self.balanced_indices)  # 
        
        # 
        final_labels = labels[self.balanced_indices]
        final_u = np.sum(final_labels == self.label_map['U'])
        final_d = np.sum(final_labels == self.label_map['D'])
        final_x = np.sum(final_labels == self.label_map['X'])
        
        print(f": U={final_u}, D={final_d}, X={final_x}, ={len(self.balanced_indices)}")
    
    def __len__(self):
        return len(self.balanced_indices)
    
    def _populate_state_dict(self, idx):
        # 
        original_idx = self.balanced_indices[idx]
        
        # 
        if hasattr(self.dataset, "get_sample"):
            data, metadata = self.dataset.get_sample(original_idx)
        else:
            data, metadata = self.dataset[original_idx]
        
        # （MetadataToLabel）
        # ，
        if self.label_key not in metadata:
            # 
            raw_label = metadata.get(self.label_key, -1)
            metadata[self.label_key] = raw_label
        
        # state_dict
        # ：X(data, metadata)，MetadataToLabelmetadata
        state_dict = {
            "X": (data, metadata)
            # label，MetadataToLabelX
        }
        
        return state_dict
