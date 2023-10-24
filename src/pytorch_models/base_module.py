from torch import optim
from pytorch_lightning import LightningModule
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from torchmetrics import Metric, MetricCollection
from pytorch_models.metrics_at_k import *
from pprint import pprint
from typing import Type
from collections.abc import Mapping


class BaseModule(LightningModule):
    """
    Base LightningModule that configures learning for modules with dense and sparse gradients.
    """

    def __init__(
        self,
        learning_rate: float = 1e-2,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-8,
        scheduler: str = "constant",  # "linear" or "constant" or "cosine" or "cosine_restart"
        warmup_steps: float = 0.0,  # from 0-1 for % of training steps, >1 for number of steps
        num_cycles: int = 1,  # Only used for "cosine" and "cosine_restart"
        dense_optim_class: Type[
            optim.Optimizer
        ] = optim.AdamW,  # optim.AdamW or optim.Adam are recommended
        sparse_optim_class: Type[optim.Optimizer] = optim.SparseAdam,
        metric_dict: Mapping[str, Metric] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if metric_dict is None:  # Default metrics for now
            metric_dict = {f"p@{i}": PrecisionAtK(top_k=i) for i in range(1, 6)}
            # metric_dict.update({f"r@{i}": RecallAtK(top_k=i) for i in range(1, 6)})
        self.metrics = MetricCollection(metric_dict)

    def set_metrics(self, metric_dict: Mapping[str, Metric]) -> None:
        self.metrics = MetricCollection(metric_dict)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return

        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        if isinstance(self.trainer.num_devices, (tuple, list)):
            num_devices = len(self.trainer.num_devices)
        else:
            num_devices = max(1, self.trainer.num_devices)

        tb_size = train_loader.batch_size * num_devices
        ab_size = self.trainer.accumulate_grad_batches
        dl_size = len(train_loader.dataset) * self.trainer.max_epochs
        self.total_steps = dl_size // tb_size // ab_size

        self.num_warmup_steps = self.hparams.warmup_steps
        if self.hparams.warmup_steps < 1:
            self.num_warmup_steps = int(self.total_steps * self.hparams.warmup_steps)

        print("Warmup_steps:", self.num_warmup_steps)
        print("Total steps:", self.total_steps)

    def _get_scheduler(
        self, optimizer, num_warmup_steps=None, num_trainig_steps=None, num_cycles=None
    ):
        available_scheduler_getters = {
            "constant": get_constant_schedule_with_warmup,
            "linear": get_linear_schedule_with_warmup,
            "cosine": get_cosine_schedule_with_warmup,
            "cosine_restart": get_cosine_with_hard_restarts_schedule_with_warmup,
        }
        scheduler_getter = available_scheduler_getters[self.hparams.scheduler]

        scheduler_args = {}
        getter_supported_args = scheduler_getter.__code__.co_varnames
        if "num_warmup_steps" in getter_supported_args:
            scheduler_args["num_warmup_steps"] = num_warmup_steps
        if "num_training_steps" in getter_supported_args:
            scheduler_args["num_training_steps"] = num_trainig_steps
        if "num_cycles" in getter_supported_args:
            scheduler_args["num_cycles"] = num_cycles

        return scheduler_getter(optimizer, **scheduler_args)

    def _get_optimizer_and_scheduler(self, optim_cls, params):
        optim_args = {"lr": self.hparams.learning_rate}
        optim_supported_args = optim_cls.__init__.__code__.co_varnames
        if "eps" in optim_supported_args:
            optim_args["eps"] = self.hparams.adam_epsilon
        if "weight_decay" in optim_supported_args:
            optim_args["weight_decay"] = self.hparams.weight_decay
        optimizer = optim_cls(params, **optim_args)
        scheduler = self._get_scheduler(
            optimizer, self.num_warmup_steps, self.total_steps, self.hparams.num_cycles
        )

        return optimizer, scheduler

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        dense_params = self.dense_optimizer_parameters()
        sparse_params = self.sparse_optimizer_parameters()

        optimizers = []
        schedulers = []

        if len(dense_params):
            o, s = self._get_optimizer_and_scheduler(
                self.hparams.dense_optim_class, dense_params
            )
            optimizers.append(o)
            schedulers.append(s)

        if len(sparse_params):
            o, s = self._get_optimizer_and_scheduler(
                self.hparams.sparse_optim_class, sparse_params
            )
            optimizers.append(o)
            schedulers.append(s)

        print(optimizers, schedulers)
        return optimizers, schedulers

    def dense_optimizer_parameters(self):
        return list(self.parameters())

    def sparse_optimizer_parameters(self):
        return []

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (tuple, list)):
            optimizers = [optimizers]
        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, (tuple, list)):
            lr_schedulers = [lr_schedulers]

        for opt in optimizers:
            opt.zero_grad()

        # Modules should always return a (loss, output) in `forward` method if target is provided
        loss, _ = self.forward(**batch)
        self.manual_backward(loss)

        for opt in optimizers:
            opt.step()

        for sch in lr_schedulers:
            sch.step()

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def train_log(self):
        pass

    def _eval_step(self, batch, batch_idx, step_name="eval"):
        loss, output = self.forward(**batch)

        if self.metrics is not None:
            self.log(
                f"{step_name}_performance",
                self.metrics(output, batch["target_ids"]),
                on_epoch=True,
                logger=True,
            )

        self.log(f"{step_name}_loss", loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, step_name="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(**batch)
        return output

    def validation_epoch_end(self, outputs):
        if self.metrics is not None:
            print("Validation performance:")
            pprint(self.metrics.compute())
            self.metrics.reset()
