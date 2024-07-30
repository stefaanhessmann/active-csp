import os
import subprocess
from typing import Optional, Dict
import uuid
import yaml
import time

__all__ = ["HardwareInterface", "SlurmInterface", "LocalInterface"]


class HardwareInterface:

    def submit_job(
        self,
        executable: Optional[str] = None,
        script: Optional[str] = None,
        arguments_string: Optional[str] = None,
        slurm_args: Optional[Dict] = None,
        wrapper_script: Optional[str] = "slurm_job.sh",
        delimiter: str = "=",
    ) -> int:
        raise NotImplementedError

    def get_state(self, job_id: int) -> str:
        raise NotImplementedError

    def job_failed(self, job_id: int) -> int:
        raise NotImplementedError


class SlurmInterface(HardwareInterface):

    def submit_job(
        self,
        executable: Optional[str] = None,
        script: Optional[str] = None,
        arguments_string: Optional[str] = None,
        slurm_args: Optional[Dict] = None,
        wrapper_script: Optional[str] = "slurm_job.sh",
        delimiter: str = "=",
    ):
        # remove wrapper script if needed
        if os.path.exists(wrapper_script):
            os.remove(wrapper_script)

        # write wrapper script
        with open(wrapper_script, "w") as fh:
            # slurm header
            fh.write("#!/bin/bash\n")
            for k, v in slurm_args.items():
                fh.write(f"#SBATCH --{k}={v}\n")

            # slurm job
            job_command = r""
            if executable is not None:
                job_command += rf"{executable} "
            if script is not None:
                job_command += rf"{script} "
            if arguments_string is not None:
                job_command += rf"{arguments_string} "

            # apptainer will remove double quotes. use \" instead
            if job_command.startswith("apptainer"):
                job_command = job_command.replace(r'"', r"\"")

            fh.write(f"\n")
            fh.write(job_command)

        # submit job
        process = subprocess.Popen(
            ["sbatch", wrapper_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        print(stdout, stderr)
        job_id = int(stdout.decode().strip().split()[-1])

        return job_id

    def get_state(self, job_id):
        # if the job just moved from pending to running, state might be ""
        state = ""
        while state == "":

            # Run the sacct command to get the job state
            command = f"sacct -X -j {job_id} --format=State --noheader"
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Check if the job ID is valid and return the state
            state = result.stdout.strip()
        return state

    def job_failed(self, job_id):
        job_state = self.get_state(job_id)
        if job_state in ["RUNNING", "COMPLETED", "SUSPENDED"]:
            return False
        return True


class LocalInterface(HardwareInterface):

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = os.path.abspath(log_dir)
        self.processes_log = os.path.join(self.log_dir, "processes.yaml")
        self.processes = dict()
        if os.path.exists(self.processes_log):
            with open(self.processes_log, "r") as file:
                self.processes = yaml.full_load(file)
        self.task_idx = 0

    def submit_job(
        self,
        executable: Optional[str] = None,
        script: Optional[str] = None,
        arguments_string: Optional[str] = None,
        slurm_args: Optional[Dict] = None,
        wrapper_script: Optional[str] = "slurm_job.sh",
        delimiter: str = "=",
    ):
        if slurm_args is None:
            slurm_args = dict()
        log_idx = str(uuid.uuid1())

        # remove wrapper script if needed
        if os.path.exists(wrapper_script):
            os.remove(wrapper_script)

        # write wrapper script
        with open(wrapper_script, "w") as fh:
            # slurm header
            fh.write("#!/bin/bash\n")

            # slurm job
            job_command = ""
            if executable is not None:
                job_command += f"{executable} "
            if script is not None:
                job_command += f"{script} "
            if arguments_string is not None:
                job_command += f"{arguments_string} "
            fh.write(f"\n")
            print(f"echo . >> {self._get_start_file(log_idx)}\n")
            fh.write(f"echo . >> {self._get_start_file(log_idx)}\n")
            fh.write(
                f"{job_command} && mv {self._get_start_file(log_idx)} {self._get_done_file(log_idx)} || "
                f"mv {self._get_start_file(log_idx)} {self._get_failed_file(log_idx)}"
            )
            time.sleep(3)

        # submit job
        log_path = "log.txt"
        if "log_path" in slurm_args.keys():
            log_path = slurm_args["log_path"]
            log_path = log_path.replace("%idx", f"{self.task_idx}")
        with open(log_path, "w") as file:
            process = subprocess.Popen(
                ["bash", wrapper_script], stdout=file, stderr=file
            )
        job_id = process.pid

        self.processes[job_id] = log_idx

        with open(self.processes_log, "w") as file:
            yaml.dump(self.processes, file)

        self.task_idx += 1
        return job_id

    def _get_start_file(self, log_idx):
        return os.path.join(self.log_dir, log_idx + ".started")

    def _get_done_file(self, log_idx):
        return os.path.join(self.log_dir, log_idx + ".done")

    def _get_failed_file(self, log_idx):
        return os.path.join(self.log_dir, log_idx + ".failed")

    def _is_started(self, job_id):
        return os.path.exists(self._get_start_file(self.processes[job_id]))

    def _is_done(self, job_id):
        return os.path.exists(self._get_done_file(self.processes[job_id]))

    def _is_failed(self, job_id):
        return os.path.exists(self._get_failed_file(self.processes[job_id]))

    def _pid_running(self, job_id):
        try:
            # Sending signal 0 doesn't send a signal, but raises an OSError if the process doesn't exist
            os.kill(job_id, 0)
            return True
        except OSError:
            return False

    def get_state(self, job_id):
        if self._is_done(job_id):
            return "COMPLETED"
        elif self._is_started(job_id):
            return "RUNNING"
        elif self._is_failed(job_id):
            return "FAILED"
        else:
            raise NotImplementedError("invalid state")
