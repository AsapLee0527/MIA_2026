"""Dataset adapters for the connectome modeling pipeline."""

from .connectome_dataset import ConnectomeDataset, group_stratified_kfold

__all__ = ["ConnectomeDataset", "group_stratified_kfold"]
