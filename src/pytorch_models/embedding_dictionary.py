import torch
from torch import nn


class XT2EmbeddingBag(nn.Module):
    """
    Implementation of torch.nn.EmbeddingBag developed for validation/testing purposes.
    """
    def __init__(self, dict_size, embed_dim, padding_idx=0, mode="mean", sparse=False):
        super().__init__()
        self.dict_size = dict_size
        self.embed_dim = embed_dim
        if mode not in ["sum", "mean"]:
            raise ValueError(f"Not supported mode type {mode}")
        self.mode = mode
        self.padding_idx = padding_idx
        self.sparse = sparse

        self.embeddings = nn.Embedding(self.dict_size + 1, self.embed_dim, padding_idx=padding_idx, sparse=sparse)
        nn.init.xavier_uniform_(self.embeddings.weight.data)
        self.embeddings.weight.data[padding_idx] = torch.zeros(self.embed_dim)

    def forward(self, lookup_tensor, per_sample_weights=None):
        output = self.embeddings(lookup_tensor)

        if per_sample_weights is not None:
            batch_dim, vec_dim, emed_dim = output.shape
            # Naive, slow version
            # for i in range(batch_dim):
            #     output[i] = output[i] * per_sample_weights[i].view(vec_dim, 1)

            # Effective version
            output = output * per_sample_weights.unsqueeze(2)

        if self.mode == "sum":
            output = output.sum(dim=1)
        elif self.mode == "mean":
            output = output.mean(dim=1)

        return output


class EmbeddingDictionary(nn.Module):
    """
    Embedding dictionary that supports weighted average.
    """
    def __init__(self, dict_size, embed_dim, mode="mean", padding_idx=0, sparse=False, EmbeddingBagClass=nn.EmbeddingBag):
        super().__init__()
        self.dict_size = dict_size
        self.embed_dim = embed_dim
        if mode not in ["sum", "mean"]:
            raise ValueError(f"Not supported mode type {mode}")
        self.mode = mode
        self.embedding_bag = EmbeddingBagClass(self.dict_size + 1, self.embed_dim, padding_idx=0, mode="sum", sparse=sparse)
        nn.init.xavier_uniform_(self.embedding_bag.weight.data)
        self.embedding_bag.weight.data[padding_idx] = torch.zeros(self.embed_dim)

    def forward(self, lookup_tensor, weights_tensor=None):
        output = self.embedding_bag(lookup_tensor, per_sample_weights=weights_tensor)

        if self.mode == "mean" and weights_tensor is not None:  # Weighted average
            output = output / weights_tensor.sum(dim=1).view(-1, 1)
        return output
