from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict
from filelock import FileLock
import yaml
import time
import os


__all__ = ["StatusTracker"]


def dump_yaml(
    data: Dict,
    file_path: str,
) -> None:
    if not file_path.endswith(".yaml"):
        raise Exception(f"{file_path} is not a valid yaml file.")
    with FileLock(file_path + ".lock"):
        with open(file_path, "w") as file:
            yaml.dump(data, file)


class StatusTracker(Callback):

    def __init__(
        self,
        done_path: str,
        status_path: Optional[str] = None,
    ):
        self.start_time = time.time()
        self.done_path = done_path
        self.status_path = status_path

        if self.status_path is not None:
            dump_yaml(
                dict(
                    job_id=int(os.getenv("SLURM_JOB_ID")),
                    start_time=self.start_time,
                    last_update=time.time(),
                ),
                self.status_path,
            )

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.status_path is not None:
            dump_yaml(
                dict(
                    job_id=int(os.getenv("SLURM_JOB_ID")),
                    start_time=self.start_time,
                    last_update=time.time(),
                ),
                self.status_path,
            )

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.status_path is not None:
            dump_yaml(
                dict(
                    job_id=int(os.getenv("SLURM_JOB_ID")),
                    start_time=self.start_time,
                    last_update=time.time(),
                ),
                self.status_path,
            )

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        with open(self.done_path, "w") as file:
            file.write(".")
