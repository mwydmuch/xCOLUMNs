import torch
from torchmetrics import Metric


class MetricAtK(Metric):
    def __init__(self, top_k: int = 1, dist_sync_on_step: bool = False, dense_pred: bool = True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.top_k = top_k
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.dense_pred = dense_pred

    @staticmethod
    def _check_tensor(tensor: torch.Tensor):
        if tensor.dim() == 1:
            tensor = tensor.reshape(tensor.shape[0], 1)
        return tensor

    def _tp_at_k(self, pred: torch.Tensor, target_ids: torch.Tensor):
        pred = MetricAtK._check_tensor(pred)
        target_ids = MetricAtK._check_tensor(target_ids)

        if self.dense_pred:
            top_k_pred = torch.argsort(pred, dim=1, descending=True)[:, : self.top_k]
        else:
            top_k_pred = pred[:, : self.top_k]
        tp_at_k = torch.zeros(target_ids.shape[0], dtype=torch.float, device=self.device)
        for i in range(self.top_k):
            tp_at_k += (target_ids == top_k_pred[:,i].unsqueeze(1)).sum(dim=1)
        return tp_at_k

    def compute(self):
        return self.sum / self.count


class PrecisionAtK(MetricAtK):
    def __init__(self, top_k: int = 1, dist_sync_on_step: bool = False, dense_pred: bool = True):
        super().__init__(top_k=top_k, dist_sync_on_step=dist_sync_on_step, dense_pred=dense_pred)

    def update(self, pred: torch.Tensor, target_ids: torch.Tensor):
        tp_at_k = self._tp_at_k(pred, target_ids)
        self.sum += (tp_at_k / self.top_k).sum()
        self.count += target_ids.shape[0]


class RecallAtK(MetricAtK):
    def __init__(self, top_k: int = 1, dist_sync_on_step: bool = False, dense_pred: bool = True, eps=1e-8):
        super().__init__(top_k=top_k, dist_sync_on_step=dist_sync_on_step, dense_pred=dense_pred)
        self.eps = eps

    def update(self, pred: torch.Tensor, target_ids: torch.Tensor):
        tp_at_k = self._tp_at_k(pred, target_ids)
        self.sum += (tp_at_k / ((target_ids > 0).sum(dim=1) + self.eps)).sum()
        self.count += target_ids.shape[0]
