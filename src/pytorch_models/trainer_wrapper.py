from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TrainerWrapper:
    def __init__(
        self,
        monitor="val_loss",
        ckpt_dir=None,
        trainer_args=None,
        early_stopping: bool = False,
        early_stopping_args=None,
        model_checkpoint: bool = True,
        model_checkpoint_args=None,
        callbacks=None,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose

        if callbacks is None:
            callbacks = []

        # Checkpointing settings
        if model_checkpoint:
            self.checkpoint_args = {
                "filename": "{epoch}-{" + monitor + ":.5f}",
                "monitor": monitor,
                "mode": "min",
                "save_top_k": 3,
                "verbose": verbose,
            }
            if model_checkpoint_args:
                self.checkpoint_args.update(model_checkpoint)
            callbacks.append(ModelCheckpoint(self.ckpt_dir, **self.checkpoint_args))

        # Early stopping settings
        if early_stopping:
            self.early_stopping_args = {
                "monitor": monitor,
                "min_delta": 0.001,
                "patience": 3, # 10
                "mode": "min",
                "verbose": verbose,
            }
            if early_stopping_args:
                self.early_stopping_args.update(early_stopping_args)
            callbacks.append(EarlyStopping(**self.early_stopping_args))

        # Trainer args
        self.trainer_args = {
            "accelerator": "gpu",
            "devices": 1,
            "precision": 32,
            "callbacks": callbacks,
            # "weights_summary": "top" if verbose else None,
            "enable_progress_bar": verbose,
            # "profiler": "simple"
        }
        if trainer_args:
            self.trainer_args.update(trainer_args)

        self.trainer = Trainer(**self.trainer_args)

    def fit(self, module, *args, **kwargs):
        if self.verbose:
            print(f"Starting training {module.__class__.__name__} ...")

        self.trainer.fit(module, *args, **kwargs)

    def test(self, module, *args, **kwargs):
        if self.verbose:
            print(f"Starting testing {module.__class__.__name__} ...")

        self.trainer.test(module, *args, **kwargs)

    def predict(self, module, *args, **kwargs):
        if self.verbose:
            print(f"Starting predicting with {module.__class__.__name__} ...")

        return self.trainer.predict(module, *args, **kwargs)

    def save_checkpoint(self, chkp_path):
        self.trainer.save_checkpoint(chkp_path)
