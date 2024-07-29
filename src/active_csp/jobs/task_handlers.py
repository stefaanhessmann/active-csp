import os
import shutil

import yaml
from ase.db import connect
import numpy as np
from ase.io.cif import write_cif
from ase.io import read
from deepcryspy.utils import read_yaml, split_ids
from collections import defaultdict
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from deepcryspy.utils.state_tracker import StateTracker
from deepcryspy import Paths
import json
from deepcryspy.job_handlers.hardware_interface import HardwareInterface
from deepcryspy import Properties
import time
import subprocess


__all__ = [
    "TaskHandler",
    "TrainHandler",
    "ComputationHandler",
    "OptimizationHandler",
    "PoolOptimizationHandler",
    "RepresentationComputationHandler",
    "StoppingCriterion",
]


class TaskHandler:

    _job_state_file = "state.yaml"

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        executable: Optional[str] = None,
        script: Optional[str] = None,
        slurm_args: Optional[Dict] = None,
        base_work_dir: Optional[str] = None,
        base_fin_dir: Optional[str] = None,
        state_path: str = "state.yaml",
    ):
        self.hardware_interface = hardware_interface
        self.executable = executable
        self.script = script
        self.slurm_args = slurm_args or dict()
        self.base_work_dir = os.path.abspath(os.path.join(base_work_dir))
        self.base_fin_dir = os.path.abspath(os.path.join(base_fin_dir))

        # create paths
        if self.base_work_dir is not None:
            os.makedirs(self.base_work_dir, exist_ok=True)
        if self.base_fin_dir is not None:
            os.makedirs(self.base_fin_dir, exist_ok=True)

        # load state
        self.state_path = os.path.abspath(state_path)
        self.state = StateTracker(
            state_path=self.state_path,
            initial_state=dict(
                cycle=-1,
                _task_idx=0,
                jobs=dict(),
            ),
        )
        self.cycle = None
        self.jobs = None
        self._task_idx = None
        self._load_state()

    def jobs_running(self):
        return len(self.jobs) != 0

    def handle_jobs(self):
        # nothing to do if no jobs are running
        if len(self.jobs) == 0:
            return

        # loop over jobs and check job status
        jobs_updated = dict()
        for task_idx, job_idx in self.jobs.items():
            job_state = self.hardware_interface.get_state(job_id=job_idx)

            # check if job is done
            if job_state == "COMPLETED":
                self._handle_finished(task_idx)
            # still running
            elif job_state in ["RUNNING", "QUEUEING", "PENDING", "SUSPENDED"]:
                jobs_updated[task_idx] = job_idx
            # check for fails
            else:
                job_idx = self._handle_failed(task_idx)
                if job_idx is not None:
                    jobs_updated[task_idx] = job_idx
        self._update_state(jobs=jobs_updated)

        # tear down after cycle
        if len(jobs_updated) == 0:
            self._on_batch_end()

    def _on_batch_end(self):
        pass

    def _handle_finished(self, idx):
        raise NotImplementedError

    def _resubmit(self, idx):
        raise NotImplementedError

    def _handle_failed(self, idx) -> Optional[int]:
        raise NotImplementedError

    def _update_job_state(self, idx, state):
        self.jobs[idx] = state
        self._update_state(jobs=self.jobs)

    def _remove_job(self, idx):
        jobs = {k: v for k, v in self.jobs.items() if k != idx}
        self._update_state(jobs=jobs)

    def _update_state(
        self,
        cycle=None,
        _task_idx=None,
        jobs=None,
    ):
        if cycle is not None:
            self.cycle = cycle
        if _task_idx is not None:
            self._task_idx = _task_idx
        if jobs is not None:
            self.jobs = jobs
        self.state.update_state(
            cycle=self.cycle,
            _task_idx=self._task_idx,
            jobs=self.jobs,
        )

    def _load_state(self):
        state = self.state.get_state()
        self.cycle = state["cycle"]
        self.jobs = state["jobs"]
        self._task_idx = state["_task_idx"]


class TrainHandler(TaskHandler):

    _done_file = ".done"

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        n_models: int,
        experiment: str,
        db_path: str,
        cutoff: float,
        n_atom_basis: int,
        n_interactions: int,
        l_max: int,
        lr_patience: int,
        lr_factor: float,
        early_stopping_patience: int,
        inference_path: str,
        configs_path: str,
        accelerator: str = "auto",
        executable: Optional[str] = None,
        script: Optional[str] = None,
        slurm_args: Optional[Dict] = None,
        state_path: str = "train_state.yaml",
        filter_data: bool = True,
    ):
        self.n_models = n_models
        self.experiment = experiment
        self.db_path = db_path
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.early_stopping_patience = early_stopping_patience
        self.cutoff = cutoff
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.l_max = l_max
        self.inference_path = inference_path
        self.configs_path = configs_path
        self.accelerator = accelerator
        self.filter_data = filter_data

        super(TrainHandler, self).__init__(
            hardware_interface,
            executable,
            script,
            slurm_args,
            os.path.join(Paths.work_dir, Paths.train_dir),
            os.path.join(Paths.fin_dir, Paths.train_dir),
            state_path,
        )
        self.splits_path = os.path.join(self.base_work_dir, "splits.npz")

    def submit_jobs(self):
        self._create_splits()
        for _ in range(self.n_models):
            job_id = self._submit_job(self._task_idx)
            self._update_job_state(self._task_idx, job_id)
            self._update_state(_task_idx=self._task_idx + 1)
        self._update_state(cycle=self.cycle + 1)
        time.sleep(60)

    def _get_run_id(self, model_idx):
        return f"model_{model_idx}"

    def _get_work_dir(self, model_idx):
        return os.path.join(self.base_work_dir, self._get_run_id(model_idx))

    def _get_fin_dir(self, model_idx):
        return os.path.join(self.base_fin_dir, self._get_run_id(model_idx))

    def _submit_job(self, task_idx):
        # todo: fix configs path
        arguments_string = (
            f"--config-dir={self.configs_path} "
            f"experiment={self.experiment} "
            f"data.datapath={os.path.join(os.path.abspath('.'), self.db_path)} "
            f"run.id={self._get_run_id(task_idx)} "
            f"run.path={self.base_work_dir} "
            f"+data.split_file={self.splits_path} "
            f"task.scheduler_args.patience={self.lr_patience} "
            f"task.scheduler_args.factor={self.lr_factor} "
            f"callbacks.early_stopping.patience={self.early_stopping_patience} "
            f"model.representation.n_atom_basis={self.n_atom_basis} "
            f"model.representation.n_interactions={self.n_interactions} "
            f"model.representation.lmax={self.l_max} "
            f"trainer.accelerator={self.accelerator} "
            f"globals.cutoff={self.cutoff}"
        )
        job_id = self.hardware_interface.submit_job(
            executable=self.executable,
            script=self.script,
            arguments_string=arguments_string,
            slurm_args=self.slurm_args,
        )
        return job_id

    def _add_model_to_ensemble(self, model_path):
        if os.path.exists(self.inference_path):
            models = torch.load(self.inference_path)
        else:
            models = nn.ModuleList()
        new_model = torch.load(model_path, map_location="cpu")
        models.append(new_model)
        torch.save(models, self.inference_path)

    def _create_splits(self):
        if self.filter_data:
            ids = self._filter_ids()
        else:
            with connect(self.db_path) as db:
                ids = list(range(len(db)))
        train_ids, val_ids, test_ids = split_ids(
            ids=ids,
            n_train=0.8,
            n_val=0.2,
            n_test=0.0,
        )
        np.savez(
            self.splits_path,
            train_idx=train_ids,
            val_idx=val_ids,
            test_idx=test_ids,
        )

    def _filter_ids(self):
        # compute stats
        f_max, s_max = [], []
        with connect(self.db_path) as db:
            n_data = len(db)
            for idx in range(n_data):
                try:
                    atmsrw = db.get(idx + 1)
                    data = atmsrw.data
                except KeyError:
                    continue
                if not Properties.f in data.keys():
                    continue

                f_max.append(np.max(np.linalg.norm(data[Properties.f], axis=1)))
                s_max.append(np.max(np.linalg.norm(data[Properties.s], axis=1)))

        f_mean = np.mean(f_max)
        s_mean = np.mean(s_max)
        f_std = np.std(f_max)
        s_std = np.std(s_max)

        # filter ids
        filtered_ids = []
        for idx, (f, s) in enumerate(zip(f_max, s_max)):
            if f <= f_mean + 3 * f_std and s <= s_mean + 3 * s_std and f <= 500:
                filtered_ids.append(idx)

        return filtered_ids

    def _get_stats(db_path):
        f_max, s_max = [], []
        with connect(db_path) as db:
            n_data = len(db)
            for idx in range(n_data):
                try:
                    atmsrw = db.get(idx + 1)
                    data = atmsrw.data
                except KeyError:
                    continue
                if not Properties.f in data.keys():
                    continue

                f_max.append(np.max(np.abs(data[Properties.f])))
                s_max.append(np.max(np.abs(data[Properties.s])))
        return {Properties.f: np.mean(f_max), Properties.s: np.mean(s_max)}, {
            Properties.f: np.std(f_max),
            Properties.s: np.std(s_max),
        }

    def _on_batch_end(self):
        os.rename(
            self.splits_path,
            self.splits_path.replace(".npz", f"_{self.cycle}.npz"),
        )

    def _handle_failed(self, task_idx):
        job_idx = self._submit_job(task_idx)
        return job_idx

    def _handle_finished(self, task_idx):
        self._add_model_to_ensemble(
            os.path.join(self._get_work_dir(task_idx), "best_model"),
        )
        shutil.move(self._get_work_dir(task_idx), self._get_fin_dir(task_idx))


class ComputationHandler(TaskHandler):

    _input_file = "input.cif"

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        executable: str,
        script: str,
        calculator_inputs: Dict,
        kspacing: float,
        pseudopotentials: Dict,
        pwx_path: str,
        n_threads: int,
        n_processes: int,
        results_db_path: str,
        slurm_args: Optional[Dict] = None,
        state_path="sp_dft_state.yaml",
        property_unit_dict: Optional[Dict] = None,
        distance_unit: str = "Ang",
        work_dir: Optional[str] = os.path.join(
            Paths.work_dir, Paths.reference_computation_dir
        ),
        fin_dir: Optional[str] = os.path.join(
            Paths.fin_dir, Paths.reference_computation_dir
        ),
    ):
        # todo: use callbacks for storing results
        self.results_db_path = os.path.abspath(results_db_path)

        if not os.path.exists(self.results_db_path):
            self._initialize_db(property_unit_dict, distance_unit)
        self.calculator_inputs = {k: v for k, v in calculator_inputs.items()}
        self.kspacing = kspacing
        self.pseudopotentials = {k: v for k, v in pseudopotentials.items()}
        self.pwx_path = pwx_path
        self.n_threads = n_threads
        self.n_processes = n_processes
        super(ComputationHandler, self).__init__(
            hardware_interface,
            executable,
            script,
            slurm_args,
            work_dir,
            fin_dir,
            state_path,
        )

    def _initialize_db(self, property_unit_dict, distance_unit):
        """
        Create database and initialize with units metadata.
        """
        if property_unit_dict is None:
            property_unit_dict = {
                Properties.e: "eV",
                Properties.f: "eV/Ang",
                Properties.s: "eV/Ang/Ang/Ang",
            }
        with connect(self.results_db_path) as db:
            db.metadata = dict(
                _property_unit_dict=property_unit_dict,
                _distance_unit=distance_unit,
            )

    def _get_work_dir(self, idx: int):
        return os.path.join(self.base_work_dir, Paths.idx_to_dir(idx))

    def _get_fin_dir(self, idx):
        fin_dir = os.path.join(self.base_fin_dir, Paths.idx_to_dir(idx))
        while os.path.exists(fin_dir):
            if "." not in os.path.basename(fin_dir):
                fin_dir += ".1"
            else:
                structure_tag, copy_count = os.path.basename(fin_dir).split(".")
                fin_dir = os.path.join(
                    self.base_fin_dir, structure_tag + f".{int(copy_count)+1}"
                )
        return fin_dir

    def submit_jobs(
        self,
        atoms_list,
        noise: float = 0.0,
    ):
        base_workdir = os.getcwd()

        for atms in atoms_list:
            # apply noise and wrap
            atms.positions = atms.positions + np.random.uniform(
                0, noise, atms.positions.shape
            )
            atms.wrap()

            # create working directory write structure
            work_dir = self._get_work_dir(self._task_idx)
            os.makedirs(work_dir)
            os.chdir(work_dir)
            write_cif(self._input_file, atms)

            # submit job
            arguments_string = (
                f"{self._input_file} "
                f"'{json.dumps(self.calculator_inputs)}' "
                f"{self.kspacing} "
                f"'{json.dumps(self.pseudopotentials)}' "
                f"{self.pwx_path} "
                f"--n_threads {self.n_threads} "
                f"--n_processes {self.n_processes} "
                f"--state_path {self._job_state_file}"
            )
            job_idx = self.hardware_interface.submit_job(
                executable=self.executable,
                script=self.script,
                arguments_string=arguments_string,
                slurm_args=self.slurm_args,
            )

            # move back to base working directory
            os.chdir(base_workdir)

            # update state
            self._update_job_state(self._task_idx, job_idx)
            self._update_state(_task_idx=self._task_idx + 1)

        self._update_state(
            cycle=self.cycle + 1,
        )
        time.sleep(60)

    def _handle_finished(self, idx):
        work_dir = self._get_work_dir(idx)
        results = read_yaml(os.path.join(work_dir, self._job_state_file))
        atms = read(os.path.join(work_dir, self._input_file))

        with connect(self.results_db_path) as db:
            db.write(
                atms,
                data=dict(
                    energy=np.array(results["energy"]),
                    forces=np.array(results["forces"]),
                    stress=np.array([results["stress"]]),
                    structure_idx=np.array([-1]),
                ),
            )
        shutil.rmtree(os.path.join(work_dir, "relaxation.save"))
        shutil.move(work_dir, self._get_fin_dir(idx))

    def _handle_failed(self, idx):
        work_dir = self._get_work_dir(idx)
        relaxation_folder = os.path.join(work_dir, "relaxation.save")
        if os.path.exists(relaxation_folder):
            shutil.rmtree(relaxation_folder)
        shutil.move(work_dir, self._get_fin_dir(idx))


class OptimizationHandler(TaskHandler):

    _input_file = "input.cif"
    _output_file = "output.cif"

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        executable: str,
        script: str,
        calculator_inputs: Dict,
        kspacing: float,
        pseudopotentials: Dict,
        pwx_path: str,
        n_threads: int,
        n_processes: int,
        max_steps: int,
        damping: float,
        force_th: float,
        slurm_args: Optional[Dict] = None,
        work_dir: str = os.path.join(Paths.work_dir, Paths.reference_optimization_dir),
        fin_dir: str = os.path.join(Paths.fin_dir, Paths.reference_optimization_dir),
        state_path: str = "state_reference_optimizer.yaml",
    ):
        self.calculator_inputs = {k: v for k, v in calculator_inputs.items()}
        self.kspacing = kspacing
        self.pseudopotentials = {k: v for k, v in pseudopotentials.items()}
        self.pwx_path = pwx_path
        self.n_threads = n_threads
        self.n_processes = n_processes
        self.max_steps = max_steps
        self.damping = damping
        self.force_th = force_th

        super(OptimizationHandler, self).__init__(
            hardware_interface,
            executable,
            script,
            slurm_args,
            work_dir,
            fin_dir,
            state_path,
        )

    def _get_work_dir(self, idx):
        return os.path.join(self.base_work_dir, Paths.idx_to_dir(idx))

    def _get_fin_dir(self, idx):
        fin_dir = os.path.join(self.base_fin_dir, Paths.idx_to_dir(idx))
        while os.path.exists(fin_dir):
            if "." not in os.path.basename(fin_dir):
                fin_dir += ".1"
            else:
                structure_tag, copy_count = os.path.basename(fin_dir).split(".")
                fin_dir = os.path.join(
                    self.base_fin_dir, structure_tag + f".{int(copy_count)+1}"
                )
        return fin_dir

    def submit_jobs(self, atoms_list):
        base_workdir = os.getcwd()

        for atms in atoms_list:
            atms.wrap()

            # build and move to working dir; write inputs files
            work_dir = self._get_work_dir(self._task_idx)
            os.makedirs(work_dir)
            os.chdir(work_dir)
            write_cif(self._input_file, atms)
            with open("cycle.txt", "w") as file:
                file.write(f"{self.cycle}")

            # submit job
            arguments_string = (
                f"{self._input_file} "
                f"{self._output_file} "
                f"'{json.dumps(self.calculator_inputs)}' "
                f"{self.kspacing} "
                f"'{json.dumps(self.pseudopotentials)}' "
                f"{self.pwx_path} "
                f"--n_threads {self.n_threads} "
                f"--n_processes {self.n_processes} "
                f"--max_steps {self.max_steps} "
                f"--damping {self.damping} "
                f"--force_th {self.force_th} "
                f"--state_path {self._job_state_file}"
            )
            job_idx = self.hardware_interface.submit_job(
                executable=self.executable,
                script=self.script,
                arguments_string=arguments_string,
                slurm_args=self.slurm_args,
            )

            # update state
            self._update_job_state(self._task_idx, job_idx)
            self._update_state(_task_idx=self._task_idx + 1)

            # move back to base working directory
            os.chdir(base_workdir)

        self._update_state(
            cycle=self.cycle + 1,
        )
        time.sleep(60)

    def _handle_finished(self, idx):
        work_dir = self._get_work_dir(idx)
        results = read_yaml(os.path.join(work_dir, self._job_state_file))
        # todo: maybe print or collect results
        # atms = read(os.path.join(work_dir, self._input_file))
        # relaxation_energy = results["best_value"]

        shutil.move(self._get_work_dir(idx), self._get_fin_dir(idx))

    def _handle_failed(self, idx):
        shutil.move(self._get_work_dir(idx), self._get_fin_dir(idx))


class PoolOptimizationHandler(TaskHandler):
    # todo: get the pool paths right

    _job_input_db = "input.db"
    _job_output_db = "output.db"

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        executable: str,
        script: str,
        model_path: str,
        cutoff: float,
        max_steps: int,
        force_th: float,
        energy_patience: int,
        uncertainty_th: float,
        device: str,
        n_tasks: int,
        input_db: str,
        output_db: str,
        slurm_args: Optional[Dict] = None,
        state_path="pool_optimization_state.yaml",
    ):
        self.input_db = os.path.abspath(input_db)
        self.output_db = os.path.abspath(output_db)
        self.n_tasks = n_tasks
        self.model_path = os.path.abspath(model_path)
        self.cutoff = cutoff
        self.max_steps = max_steps
        self.force_th = force_th
        self.energy_patience = energy_patience
        self.uncertainty_th = uncertainty_th
        self.device = device
        super(PoolOptimizationHandler, self).__init__(
            hardware_interface,
            executable,
            script,
            slurm_args,
            os.path.join(Paths.work_dir, Paths.pool_optimization_dir),
            os.path.join(Paths.fin_dir, Paths.pool_optimization_dir),
            state_path,
        )

    def _get_work_dir(self, idx):
        return os.path.join(self.base_work_dir, Paths.idx_to_dir(idx))

    def _get_fin_dir(self, idx):
        return os.path.join(self.base_fin_dir, Paths.idx_to_dir(idx))

    def submit_jobs(self):
        with connect(self.input_db) as db:
            pool_size = len(db)

        # split pool
        split = defaultdict(list)
        for i in range(pool_size):
            split[i % self.n_tasks].append(i)

        # submit jobs
        base_workdir = os.getcwd()
        for _, pool_ids in split.items():
            # load structures for optimization
            snippet_atoms = []
            atmsrw_metadata = []
            with connect(self.input_db) as db:
                for idx in pool_ids:
                    atmsrw = db.get(idx + 1)
                    snippet_atoms.append(atmsrw.toatoms())
                    atmsrw_metadata.append(atmsrw.key_value_pairs)

            # build task dir and move
            work_dir = self._get_work_dir(self._task_idx)
            os.makedirs(work_dir)
            os.chdir(work_dir)

            # write new db
            with connect(self._job_input_db) as input_db:
                for atms, key_value_pairs in zip(snippet_atoms, atmsrw_metadata):
                    input_db.write(atms, key_value_pairs=key_value_pairs)

            # submit job
            arguments_string = (
                f"{self.model_path} "
                f"{self._job_input_db} "
                f"{self._job_output_db} "
                f"--cutoff {self.cutoff} "
                f"--max_steps {self.max_steps} "
                f"--force_th {self.force_th} "
                f"--energy_patience {self.energy_patience} "
                f"--uncertainty_th {self.uncertainty_th} "
                f"--device {self.device} "
                f"--state_path {self._job_state_file}"
            )
            job_idx = self.hardware_interface.submit_job(
                executable=self.executable,
                script=self.script,
                arguments_string=arguments_string,
                slurm_args=self.slurm_args,
            )

            # move back to base dir
            os.chdir(base_workdir)

            # update state
            self._update_job_state(self._task_idx, job_idx)
            self._update_state(_task_idx=self._task_idx + 1)

        self._update_state(
            cycle=self.cycle + 1,
        )
        time.sleep(60)

    def _handle_failed(self, idx):
        base_workdir = os.getcwd()

        # move to task dir
        work_dir = self._get_work_dir(idx)
        os.chdir(work_dir)

        # remove old output db
        if os.path.exists(self._job_output_db):
            os.remove(self._job_output_db)

        # submit job
        arguments_string = (
            f"{self.model_path} "
            f"{self._job_input_db} "
            f"{self._job_output_db} "
            f"--cutoff {self.cutoff} "
            f"--max_steps {self.max_steps} "
            f"--force_th {self.force_th} "
            f"--energy_patience {self.energy_patience} "
            f"--uncertainty_th {self.uncertainty_th} "
            f"--device {self.device} "
            f"--state_path {self._job_state_file}"
        )
        job_idx = self.hardware_interface.submit_job(
            executable=self.executable,
            script=self.script,
            arguments_string=arguments_string,
            slurm_args=self.slurm_args,
        )

        # move back to base dir
        os.chdir(base_workdir)

        # update state
        self._update_job_state(self._task_idx, job_idx)

    def _handle_finished(self, idx):
        work_dir = self._get_work_dir(idx)
        fin_dir = self._get_fin_dir(idx)

        # copy datapoints
        with connect(os.path.join(work_dir, self._job_output_db)) as job_output_db:
            with connect(self.output_db) as output_db:
                for i in range(len(job_output_db)):
                    atmsrw = job_output_db.get(i + 1)
                    output_db.write(
                        atmsrw.toatoms(),
                        data=atmsrw.data,
                        key_value_pairs=atmsrw.key_value_pairs,
                    )

        # move folder to fin
        shutil.move(work_dir, fin_dir)


class RepresentationComputationHandler(TaskHandler):

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        executable: str,
        script: str,
        pool_path: str,
        inference_path: str,
        cutoff: float,
        batch_size: int,
        device: Optional[str] = "cuda",
        slurm_args: Optional[Dict] = None,
        state_path: str = "state_compute_representations.yaml",
    ):
        self.cutoff = cutoff
        self.batch_size = batch_size
        self.pool_path = pool_path
        self.inference_path = inference_path
        self.device = device
        super(RepresentationComputationHandler, self).__init__(
            hardware_interface,
            executable,
            script,
            slurm_args,
            os.path.join(Paths.work_dir, Paths.compute_representations_dir),
            os.path.join(Paths.fin_dir, Paths.compute_representations_dir),
            state_path,
        )

    def _get_results_file(self, cycle):
        return f"representations_{cycle}.npy"

    def handle_jobs(self):
        jobs_updated = dict()
        for task_idx, job_idx in self.jobs.items():
            work_file = os.path.join(
                self.base_work_dir, self._get_results_file(task_idx)
            )
            fin_file = os.path.join(self.base_fin_dir, self._get_results_file(task_idx))
            # hanlde finished job
            if os.path.exists(work_file):
                time.sleep(30)
                representations = np.load(work_file)
                with connect(self.pool_path) as db:
                    metadata = db.metadata
                    metadata.update(dict(representations=representations.tolist()))
                    db.metadata = metadata
                # move results file
                shutil.move(work_file, fin_file)
            else:
                jobs_updated[task_idx] = job_idx

        self._update_state(jobs=jobs_updated)

    def submit_jobs(self):
        self._submit_job()
        self._update_state(cycle=self.cycle + 1)
        time.sleep(60)

    def _submit_job(self):
        work_file = os.path.join(
            self.base_work_dir, self._get_results_file(self._task_idx)
        )
        arguments_string = (
            f"{self.inference_path} "
            f"{self.pool_path} "
            f"{work_file} "
            f"--device {self.device} "
            f"--batch_size {self.batch_size} "
            f"--cutoff {self.cutoff}"
        )
        job_id = self.hardware_interface.submit_job(
            executable=self.executable,
            script=self.script,
            arguments_string=arguments_string,
            slurm_args=self.slurm_args,
        )
        self._update_job_state(self._task_idx, job_id)
        self._update_state(_task_idx=self._task_idx + 1)


class StoppingCriterion:

    def __init__(
        self,
        hardware_interface: HardwareInterface,
        executable: Optional[str],
        script: str,
        n_models: int,
        epsilon: float,
        n_clusters: int,
        eps: float,
        min_sampled: int,
        min_different: int,
        max_tries: int,
        max_cluster_size: int,
        recluster_stable: bool,
        cutoff: float,
        batch_size: int = 10,
        device: str = "cuda",
        slurm_args: Optional[Dict] = None,
        work_dir: str = os.path.join(Paths.work_dir, "stopping_criterion"),
        fin_dir: str = os.path.join(Paths.fin_dir, "stopping_criterion"),
        include_train_structures: bool = True,
    ):
        self.cutoff = cutoff
        self.batch_size = batch_size
        self.n_models = n_models
        self.epsilon = epsilon
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_sampled = min_sampled
        self.min_different = min_different
        self.max_tries = max_tries
        self.max_cluster_size = max_cluster_size
        self.recluster_stable = recluster_stable
        self.device = device
        self.work_dir = work_dir
        self.fin_dir = fin_dir
        self.hardware_interface = hardware_interface
        self.executable = executable
        self.script = script
        self.slurm_args = slurm_args or dict()
        self.include_train_structures = include_train_structures

        # create paths
        if self.work_dir is not None:
            os.makedirs(self.work_dir, exist_ok=True)
        if self.fin_dir is not None:
            os.makedirs(self.fin_dir, exist_ok=True)

    def evaluate(self, cycle):
        work_dir = os.path.join(self.work_dir, str(cycle).zfill(2))
        os.makedirs(work_dir, exist_ok=True)
        fin_dir = os.path.join(self.fin_dir, str(cycle).zfill(2))
        stop_path = os.path.join(work_dir, "stop")
        results_path = os.path.join(work_dir, "results.yaml")
        model_path = os.path.abspath(os.path.join("data", f"inference_models_{cycle}"))
        current_db_path = os.path.abspath(os.path.join("data", f"candidate_pool.db"))
        previous_db_path = os.path.abspath(
            os.path.join("data", f"candidate_pool_{cycle}.db")
        )
        base_dir = os.getcwd()
        os.chdir(work_dir)
        arguments_string = (
            f"{model_path} "
            f"{current_db_path} "
            f"{previous_db_path} "
            f"--n_models {self.n_models} "
            f"--epsilon {self.epsilon} "
            f"--n_clusters {self.n_clusters} "
            f"--eps {self.eps} "
            f"--min_sampled {self.min_sampled} "
            f"--min_different {self.min_different} "
            f"--max_tries {self.max_tries} "
            f"--max_cluster_size {self.max_cluster_size} "
            f"--recluster_stable {self.recluster_stable} "
            f"--cutoff {self.cutoff} "
            f"--batch_size {self.batch_size} "
            f"--device {self.device} "
            f"--include_train_structures {self.include_train_structures}"
        )
        self.hardware_interface.submit_job(
            executable=self.executable,
            script=self.script,
            arguments_string=arguments_string,
            slurm_args=self.slurm_args,
        )
        os.chdir(base_dir)

        # wait for job to finish
        while not os.path.exists(stop_path):
            time.sleep(60)

        # load results
        with open(stop_path, "r") as file:
            if file.readline().strip() == "True":
                stop = True
            else:
                stop = False
        with open(results_path, "r") as file:
            ids = yaml.safe_load(file)["ids"]

        # move to fin
        shutil.move(work_dir, fin_dir)

        return stop, ids
