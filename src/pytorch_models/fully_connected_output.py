import torch
from torch import nn
import torch.nn.functional as F


class FullyConnectedOutput(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        layer_units=(),
        nonlin=nn.ReLU(),
        hidden_dropout=0,
        output_nonlin=F.sigmoid,
        loss=F.binary_cross_entropy_with_logits,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlin = nonlin
        self.layer_units = layer_units
        self.output_nonlin = output_nonlin
        self.loss = loss
        self.hidden_dropout = hidden_dropout
        self.bias = bias

        print("Initializing FullyConnectedOutput with loss function:", loss)

        sequence = []
        units = [self.input_size] + list(self.layer_units) + [self.output_size]
        for in_size, out_size in zip(units, units[1:]):
            sequence.extend(
                [
                    nn.Linear(in_size, out_size, bias=self.bias),
                    self.nonlin,
                    nn.Dropout(self.hidden_dropout),
                ]
            )

        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)
        print("Flat output, linear layers:", units)

    def forward(self, input, target=None):
        output = self.sequential(input)
        if target is not None:
            return self.loss(output, target), self.output_nonlin(output)
        else:
            return self.output_nonlin(output)
