import os
from pytorch_models.base_classifier import BaseClassifier
from pytorch_models.sparse_dataset import SparseDataset
from pytorch_models.base_data_module import BaseDataModule
from pytorch_models.baseline_modules import *
from collections.abc import Iterable


class FlatFullyConnectedClassfier(BaseClassifier):
    def __init__(self, 
                 loss,
                 hidden_units: Iterable = (),
                 learning_rate: float = 0.01, 
                 weight_decay: float = 1e-5, 
                 adam_epsilon: float = 1e-7,
                 batch_size: int = 128,
                 num_workers: int =-1,
                 negative_sampling: int = 0,
                 devices: int = 1,
                 precision: int = 16,
                 max_epochs: int = 10,
                 ckpt_dir=None):
                
        model_module_kwargs = {
            "loss": loss,
            "output_hidden_units": hidden_units,
            "learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "adam_epsilon": adam_epsilon
        }
        dataset_kwargs = {
            "input_dense_vec": True,
            "target_dense_vec": negative_sampling == 0,
            "target_negative_samples": negative_sampling,
        }
        data_module_kwargs = {
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": num_workers
        }
        trainer_wrapper_kwargs = {
            "ckpt_dir": ckpt_dir,
            "trainer_args": {"max_epochs": max_epochs, "devices": devices, "precision": precision},
            "early_stopping": False
        }
        super().__init__(FlatFullyConnectedModule, 
                         SparseDataset,
                         BaseDataModule,
                         model_module_kwargs, 
                         dataset_kwargs,
                         data_module_kwargs,
                         trainer_wrapper_kwargs)


class FlatEmbeddingClassfier(BaseClassifier):
    def __init__(self, 
                 embed_dim: int = 300, 
                 embed_mode: str = "mean",
                 hidden_units: Iterable = (),
                 learning_rate: float = 0.01, 
                 weight_decay: float = 1e-5, 
                 adam_epsilon: float = 1e-7,
                 batch_size: int = 32,
                 num_workers: int =-1,
                 negative_sampling: int = 0,
                 devices: int = 1,
                 precision: int = 16,
                 max_epochs: int = 10,
                 ckpt_dir=None):
        
        model_module_kwargs = {
            "embed_dim": embed_dim, 
            "embed_mode": embed_mode,
            "output_hidden_units": hidden_units,
            "learning_rate": learning_rate, 
            "weight_decay": weight_decay, 
            "adam_epsilon": adam_epsilon
        }
        dataset_kwargs = {
            "input_dense_vec": False,
            "target_dense_vec": negative_sampling == 0,
            "target_negative_samples": negative_sampling,
        }
        data_module_kwargs = {
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": num_workers
        }
        trainer_wrapper_kwargs = {
            "ckpt_dir": ckpt_dir,
            "trainer_args": {"max_epochs": max_epochs, "devices": devices, "precision": precision},
            "early_stopping": False
        }
        super().__init__(FlatEmbeddingModule, 
                         SparseDataset,
                         BaseDataModule,
                         model_module_kwargs, 
                         dataset_kwargs, 
                         data_module_kwargs, 
                         trainer_wrapper_kwargs)


