from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from collections.abc import Callable, Mapping
from multiprocessing import cpu_count

class BaseDataModule(LightningDataModule):
    """
    Base LightningDataModule that configures dataloaders for datasets.
    """
    def __init__(
        self,
        dataset: Mapping[str, Dataset],
        collate_fn: Callable = None,
        train_batch_size: int = 64,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        shuffle_train: bool = True,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        num_workers = num_workers if num_workers >= 0 else cpu_count() + num_workers
        self.num_workers = num_workers
        self.train_shuffle = shuffle_train
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.tokenizer = None

        print(
            f"Initializing {self} train/eval_batch_size={train_batch_size}/{eval_batch_size}, num_workers={num_workers} ..."
        )

    def _create_dataloader(self, dataset_key, batch_size=32, shuffle=False):
        dataset = self.dataset.get(dataset_key, None)
        if dataset is not None:
            collate_fn = self.collate_fn
            if collate_fn is None and hasattr(dataset, "collate_function"):
                collate_fn = dataset.collate_function()
            return DataLoader(
                self.dataset[dataset_key],
                collate_fn=collate_fn,
                batch_size=batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
            )
        else:
            return None

    def train_dataloader(self):
        return self._create_dataloader("train", batch_size=self.train_batch_size, shuffle=self.train_shuffle)

    def val_dataloader(self):
        return self._create_dataloader("val", batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return self._create_dataloader("test", batch_size=self.eval_batch_size)
    
    def predict_dataloader(self):
        return self._create_dataloader("predict", batch_size=self.eval_batch_size)
