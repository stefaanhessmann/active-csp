import torch
from torchmetrics import Metric
import numpy as np
import torch.nn as nn


__all__ = ["AngleError"]


class AngleError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("angle_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        epsilon = torch.ones_like(preds) * 1e-6

        self.angle_error += torch.sum(
            torch.arccos(self.cos(preds + epsilon, target + epsilon) * 0.99999)
        )
        self.total += np.prod(preds.shape[:-1])

    def compute(self):
        return self.angle_error / self.total
