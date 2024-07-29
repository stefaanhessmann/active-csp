import torch.nn as nn
import torch
from typing import Optional, List
from pytorch_lightning import LightningModule
from copy import deepcopy


__all__ = ["NNEnsemble"]


class NNEnsemble(LightningModule):
    def __init__(
        self,
        models: nn.ModuleList,
        properties: List[str],
        has_val_labels: bool = True,
    ):
        super(NNEnsemble, self).__init__()
        self.models = models
        if type(properties) == str:
            properties = [properties]
        self.properties = properties
        self.has_val_labels = has_val_labels

    def setup(self, stage: Optional[str] = None) -> None:
        for model in self.models:
            model.setup(stage)

    def forward(
        self,
        x,
    ):
        results = {}
        for p in self.properties:
            results[p] = []

        for model in self.models:
            predictions = model(deepcopy(x))
            for prop, values in predictions.items():
                if prop in self.properties:
                    results[prop].append(values.detach())

        means = {}
        stds = {}
        for prop, values in results.items():
            stacked_values = torch.stack(values)
            means[prop] = stacked_values.mean(0)
            stds[prop] = stacked_values.std(0)

        return means, stds

    def log_metrics(self, batch, pred, subset):
        for om in self.models[0].outputs:
            for mname, pmetric in om.metrics.items():
                self.log(
                    f"{subset}_{om.property}_{mname}",
                    pmetric(pred[om.property].detach(), batch[om.target_name]),
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                )

    def training_step(self, batch, batch_idx, optimizer_idx):
        model = self.models[optimizer_idx]
        targets = {
            output.target_name: batch[output.target_name] for output in model.outputs
        }
        pred = model(deepcopy(batch))
        loss = model.loss_fn(pred, targets) * torch.rand(1).cuda() * 2
        self.log(f"train_loss", loss, prog_bar=False, on_epoch=True, on_step=True)
        self.log_metrics(pred, targets, f"train")

        return {"loss": loss / len(self.models), "model_outputs": pred}

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        torch.set_grad_enabled(True)
        pred, uncertainty = self(deepcopy(batch))
        if self.has_val_labels:
            targets = {
                output.target_name: batch[output.target_name]
                for output in self.models[0].outputs
            }
            self.log_metrics(pred, targets, "val")
            loss = self.models[0].loss_fn(pred, targets)
        else:
            loss = 0.0
            for v in uncertainty.values():
                loss += v.mean()
        for k, v in uncertainty.items():
            self.log(
                f"val_uncertainty_{k}",
                v.mean(),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)

        return {
            "val_loss": loss / len(self.models),
            "model_outputs": dict(mean=pred, uncertainty=uncertainty),
        }

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizers, optimconfs = [], []
        for model in self.models:
            optimizer, optimconf = model.configure_optimizers()
            optimizers.append(optimizer[0])
            optimconfs.append(optimconf[0])
        return optimizers, optimconfs

    def set_inference_mode(self, value: bool):
        for model in self.models:
            model.inference_mode = value
