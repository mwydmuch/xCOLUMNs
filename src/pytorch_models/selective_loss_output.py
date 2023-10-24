import torch
from torch import nn
import torch.nn.functional as F


class SelectiveLossOutput(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        layer_units=(),
        nonlin=nn.ReLU(),
        hidden_dropout=0,
        loss=F.binary_cross_entropy_with_logits,
        output_nonlin=F.sigmoid,
        sparse=True,
        bias=True,
        padding_idx=0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.nonlin = nonlin
        self.hidden_dropout = hidden_dropout

        self.loss = loss
        self.output_nonlin = output_nonlin
        self.layer_units = layer_units
        self.padding_idx = padding_idx

        self.embed_size = layer_units[-1] if len(layer_units) else input_size
        self.bias = bias
        if bias:  # Include bias term in embeddings (weights)
            self.embed_size += 1

        self.sequential = None
        units = [self.input_size]
        if len(layer_units):
            sequence = []
            units += list(self.layer_units)
            for in_size, out_size in zip(units, units[1:]):
                sequence.extend(
                    [
                        nn.Linear(in_size, out_size, bias=self.bias),
                        self.nonlin,
                        nn.Dropout(self.hidden_dropout),
                    ]
                )

            print("\nSequence: \n", sequence)
            self.sequential = nn.Sequential(*sequence)

        units += [self.output_size]
        self.embedding = nn.Embedding(
            self.output_size + 1,
            self.embed_size,
            padding_idx=padding_idx,
            sparse=sparse,
        )
        # nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.zeros_(self.embedding.weight.data)
        self.embedding.weight.data[padding_idx] = torch.zeros(self.embed_size)
        print(f"Initializing SelectiveLossOutput with linear layers sequence={units}")

    def _predict_last_hidden(self, x):
        # Use dense linear layers first
        if self.sequential is not None:
            x = self.sequential(x)

        if self.bias:  # Add bias column to x
            x = torch.hstack((x, torch.ones(x.shape[0], 1, device=x.device)))

        return x

    def forward(self, x, target_ids=None, target_values=None, target_mask=None):
        x = self._predict_last_hidden(x)

        if (
            target_ids is not None
            and target_values is not None
            and target_mask is not None
        ):
            # Calculate output
            target_embeddings = self.embedding(target_ids)
            output = torch.bmm(target_embeddings, x.unsqueeze(2)).squeeze(2)

            # Calculate loss
            loss = self.loss(output, target_values, weight=target_mask)
            return loss, self.output_nonlin(output)  # This output shouldn't be used
        else:
            # Else predict
            return self._predict_output(x)

    def _predict_output(self, x):
        output = torch.matmul(x, self.embedding.weight.data.t())
        return self.output_nonlin(output)

    def get_output_weights(self):
        return self.embedding.weight.data
