from typing import Optional
from pprint import pprint

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from pytorch_models.base_module import BaseModule
from pytorch_models.fully_connected_output import FullyConnectedOutput


class TransformerModule(BaseModule):
    def __init__(
        self,
        model_name_or_path: str,
        output_size: int,
        output_hidden_units: tuple = (),
        loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        layer_wise_lr: bool = False,
        layer_wise_lr_mutliplier: float = 1.1,
        freeze_transformer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        print(
            f"Initializing TransformerModule with model_name={model_name_or_path} ..."
        )

        # Use build-in classification outputs
        # if task in ["build-in-multilabel", "build-in-multiclass"]:
        #     self.config = AutoConfig.from_pretrained(
        #         model_name_or_path,
        #         finetuning_task=None,
        #         # For AutoModelForSequenceClassification
        #         hidden_dropout_prob=kwargs.get("hidden_dropout", 0),
        #         output_hidden_states=False,
        #         summary_use_proj=False,
        #         num_labels=output_size,
        #     )

        #     self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        #     self.output = None

        # Use custom loss
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, finetuning_task=None
        )
        self.transformer = AutoModel.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.output = FullyConnectedOutput(
            self.config.hidden_size,
            output_size,
            layer_units=output_hidden_units,
            hidden_dropout=kwargs.get("hidden_dropout", 0),
            output_nonlin=nn.Sigmoid(),
            loss=loss,
        )

    def forward(self, input_ids=None, attention_mask=None, target=None, **kwargs):
        transformer_output = self.transformer(
            input_ids,
            attention_mask,
        )
        transformer_output = transformer_output.last_hidden_state[:, 0, :]
        output = self.output.forward(transformer_output, target)
        return output

    def dense_optimizer_parameters(self):
        return self._get_optimizer_grouped_parameters(
            layer_wise_lr=self.hparams.layer_wise_lr,
            layer_wise_lr_mutliplier=self.hparams.layer_wise_lr_mutliplier,
        )

    def _get_optimizer_grouped_parameters(
        self, layer_wise_lr=False, layer_wise_lr_mutliplier=1.1
    ):
        # It is suggested to not use any decay for bias, LayerNorm.weight and LayerNorm.weight layers.
        no_decay = ["bias", "LayerNorm.weight"]

        if layer_wise_lr:
            optimizer_grouped_parameters = []
            for name, params in self.named_parameters():
                weight_decay = (
                    0.0
                    if any(nd in name for nd in no_decay)
                    else self.hparams.weight_decay
                )
                learning_rate = self.hparams.learning_rate

                if "embeddings" in name or "encoder" in name:
                    learning_rate /= 10

                    for i in range(0, 20):
                        if f"layer.{i}" in name:
                            learning_rate *= layer_wise_lr_mutliplier ** (i + 1)

                optimizer_grouped_parameters.append(
                    {
                        "params": params,
                        "weight_decay": weight_decay,
                        "lr": learning_rate,
                    }
                )

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        return optimizer_grouped_parameters

    def save_transformer(self, ckpt_dir):
        self.transformer.save_pretrained(ckpt_dir)
