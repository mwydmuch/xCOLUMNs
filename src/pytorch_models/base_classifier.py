from collections.abc import Mapping
from torch.utils.data import Dataset
from typing import Type
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import Metric

from pytorch_models.base_data_module import BaseDataModule
from pytorch_models.trainer_wrapper import TrainerWrapper


class BaseClassifier():
    """
    Base class for all classifiers compatible with Scikit-Learn API.
    """
    def __init__(self, 
                 module_class: Type[LightningModule], 
                 dataset_class: Type[Dataset], 
                 data_module_class: Type[LightningDataModule] = BaseDataModule,
                 module_kwargs: Mapping = {},
                 dataset_kwargs: Mapping = {},
                 data_module_kwargs: Mapping = {},
                 trainer_wrapper_kwargs: Mapping = {},
                 metrics: Mapping[str, Metric] = None):

        self.module_class = module_class
        self.dataset_class = dataset_class
        self.data_module_class = data_module_class

        self.module_kwargs = module_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.data_module_kwargs = data_module_kwargs
        self.trainer_wrapper_kwargs = trainer_wrapper_kwargs
        self.metrics = metrics

        self.module = None
        self.trainer_wrappper = None

    def _create_dataset(self, X, Y=None):
        return self.dataset_class(X, Y, **self.dataset_kwargs)

    def _check_trainer(self):
        if self.trainer_wrappper is None:
            self.trainer_wrappper = TrainerWrapper(**self.trainer_wrapper_kwargs)

    def fit(self, X_train, Y_train, X_val = None, Y_val = None, val_metrics: Mapping[str, Metric] = None):
        # Create dataset
        dataset = {"train": self._create_dataset(X_train, Y_train)}
        monitor = "train_loss"
        if X_val is not None and Y_val is not None:
            dataset["val"] = self._create_dataset(X_val, Y_val)
            monitor = "val_loss"
        data_module = self.data_module_class(dataset, **self.data_module_kwargs)

        self.module = self.module_class(dataset['train'].input_size(), dataset['train'].target_size(), **self.module_kwargs)
        
        # Set metrics
        if val_metrics is not None:
            self.module.set_measutes(val_metrics)
        elif self.metrics is not None:
            self.module.set_measutes(self.metrics)

        # Fit
        self.trainer_wrapper_kwargs["monitor"] = monitor
        self._check_trainer()
        self.trainer_wrappper.fit(self.module, datamodule=data_module)
    
    def test(self, X_test, Y_test, test_metrics: Mapping[str, Metric] = None):
        # Create test dataset
        dataset = {"test": self._create_dataset(X_test, Y_test)}
        data_module = BaseDataModule(dataset, **self.data_module_kwargs)

        # Set metrics
        if test_metrics is not None:
            self.module.set_metrics(test_metrics)
        elif self.metrics is not None:
            self.module.set_metrics(self.metrics)
        
        # Test
        self.trainer_wrappper.test(self.module, datamodule=data_module)
        
    def predict(self, X_predict):
        # Create predict dataset
        dataset = {"predict": self._create_dataset(X_predict)}
        data_module = BaseDataModule(dataset, **self.data_module_kwargs)

        # Predict
        return self.trainer_wrappper.predict(self.module, data_module)
    
    def save(self, path):
        self.trainer_wrappper.save(path)

    def load(self, path):
        self.module = self.module_class.load_from_checkpoint(path)
