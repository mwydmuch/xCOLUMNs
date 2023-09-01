import torch

from pprint import pprint

from pytorch_models.base_module import BaseModule
from pytorch_models.embedding_dictionary import EmbeddingDictionary
from pytorch_models.fully_connected_output import FullyConnectedOutput
from pytorch_models.selective_loss_output import SelectiveLossOutput


class FlatFullyConnectedModule(BaseModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_hidden_units: tuple = (),
        loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.output = FullyConnectedOutput(input_size, output_size, layer_units=output_hidden_units, loss=loss)

    def forward(self, input=None, target=None, **kwargs):
        if input is None:
            return ValueError("input cannot be None")
        output = self.output(input, target)
        return output


class FlatEmbeddingModule(BaseModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embed_dim: int = 500,
        embed_mode: str = "mean",
        output_hidden_units: tuple = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.embedding = EmbeddingDictionary(input_size, embed_dim, mode=embed_mode, sparse=True)
        self.output = FullyConnectedOutput(embed_dim, output_size, layer_units=output_hidden_units)

    def forward(self, input_ids=None, input_values=None, target=None, **kwargs):
        if input_ids is None and input_values is None:
            return ValueError("input_ids and input_values cannot be None")
        input_embeddings = self.embedding(input_ids, input_values)
        output = self.output(input_embeddings, target)
        return output

    def dense_optimizer_parameters(self):
        return list(self.output.parameters())

    def sparse_optimizer_parameters(self):
        return list(self.embedding.parameters())


class FlatSelectiveModule(BaseModule):
    def __init__(self,
                input_size: int,
                output_size: int,
                output_hidden_units: tuple = (),
                **kwargs
                ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.output = SelectiveLossOutput(input_size, output_size, sparse=True, layer_units=output_hidden_units)

    def forward(self, input=None, target_ids=None, target_values=None, target_mask=None, **kwargs):
        if input is None:
            return ValueError("input cannot be None")
        output = self.output(input, target_ids, target_values, target_mask)
        return output
    
    def _eval_step(self, batch, batch_idx, step_name="eval"):
        pred = self.forward(input_ids=batch["input_ids"], input_values=batch["input_values"])
        pred[:,0] = 0
        self.log('val_performance', 
                 self.metrics(pred, batch['target_ids'] * batch['target_values']), 
                 on_epoch=True, 
                 logger=True)

        loss, _ = self.forward(**batch)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        return loss

    def dense_optimizer_parameters(self):
        return list(self.output.sequential.parameters()) if self.output.sequential else []
        
    def sparse_optimizer_parameters(self):
        return list(self.output.embedding.parameters())
    

class FlatSelectiveEmbeddingModule(BaseModule):
    def __init__(self,
                input_size: int,
                output_size: int,
                embed_dim: int = 500,
                embed_mode: str = "mean",
                output_hidden_units: tuple = (),
                **kwargs
                ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.embedding = EmbeddingDictionary(input_size, embed_dim, mode=embed_mode, sparse=True)
        self.output = SelectiveLossOutput(embed_dim, output_size, sparse=True, layer_units=output_hidden_units)

    def forward(self, input_ids=None, input_values=None, target_ids=None, target_values=None, target_mask=None, **kwargs):
        if input_ids is None and input_values is None:
            return ValueError("input_ids and input_values cannot be None")
        input_embeddings = self.embedding(input_ids, input_values)
        output = self.output(input_embeddings, target_ids, target_values, target_mask)
        return output
    
    def _eval_step(self, batch, batch_idx, step_name="eval"):
        pred = self.forward(input_ids=batch["input_ids"], input_values=batch["input_values"])
        pred[:,0] = 0
        self.log('val_performance', 
                 self.metrics(pred, batch['target_ids'] * batch['target_values']), 
                 on_epoch=True, 
                 logger=True)

        loss, _ = self.forward(**batch)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        return loss

    def dense_optimizer_parameters(self):
        return list(self.output.sequential.parameters()) if self.output.sequential else []
        
    def sparse_optimizer_parameters(self):
        return list(self.embedding.parameters()) + list(self.output.embedding.parameters())
