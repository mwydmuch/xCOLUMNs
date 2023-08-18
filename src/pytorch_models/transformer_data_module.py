from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from pytorch_models.base_data_module import BaseDataModule


class TransformerDataModule(BaseDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path

    def setup(self, stage=None):
        print(f"Obtaining tokenizer ...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        for d in self.dataset.values():
            d.setup(self.tokenizer, self.max_seq_length)
