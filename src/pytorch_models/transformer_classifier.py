import os
from pytorch_models.base_classifier import BaseClassifier
from pytorch_models.sparse_dataset import SparseDataset
from pytorch_models.baseline_modules import *
from collections.abc import Iterable

from pytorch_models.transformer_module import TransformerModule
from pytorch_models.transformer_data_module import TransformerDataModule
from pytorch_models.text_dataset import TextDataset


class TransformerClassfier(BaseClassifier):
    def __init__(
        self,
        model_name_or_path: str,
        loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        max_seq_length: int = 512,
        hidden_units: Iterable = (),
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        adam_epsilon: float = 1e-7,
        batch_size: int = 64,
        num_workers: int = -1,
        negative_sampling: int = 0,
        devices: int = 1,
        precision: int = 16,
        max_epochs: int = 10,
        ckpt_dir=None,
    ):
        model_module_kwargs = {
            "model_name_or_path": model_name_or_path,
            "loss": loss,
            "output_hidden_units": hidden_units,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "adam_epsilon": adam_epsilon,
        }
        dataset_kwargs = {}
        data_module_kwargs = {
            "model_name_or_path": model_name_or_path,
            "max_seq_length": max_seq_length,
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": num_workers,
        }
        trainer_wrapper_kwargs = {
            "ckpt_dir": ckpt_dir,
            "trainer_args": {
                "max_epochs": max_epochs,
                "devices": devices,
                "precision": precision,
            },
            "early_stopping": True,
        }
        super().__init__(
            TransformerModule,
            TextDataset,
            TransformerDataModule,
            model_module_kwargs,
            dataset_kwargs,
            data_module_kwargs,
            trainer_wrapper_kwargs,
        )
