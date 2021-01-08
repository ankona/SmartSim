import time
from shutil import which

import numpy as np

from ...error import LauncherError, SSConfigError
from ...utils import get_logger
from ..launcher import Launcher
from ..shell import execute_async_cmd
from ..stepInfo import SlurmStepInfo
from ..taskManager import TaskManager
from .slurmCommands import sacct, scancel, sstat
from .slurmParser import parse_sacct, parse_sstat_nodes, parse_step_id_from_sacct
from .slurmStep import SlurmStep

logger = get_logger(__name__)


class SlurmLauncher(Launcher):
    """This class encapsulates the functionality needed
    to manager allocations and launch jobs on systems that use
    Slurm as a workload manager.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a SlurmLauncher"""
        super().__init__(*args, **kwargs)
        self.task_manager = TaskManager()

    def create_step(self, entity_name, run_settings, multi_prog=False):
        """Convert a smartsim entity run_settings into a Slurm step

        This function convert a smartsim entity run_settings
        into a Slurm stepto be launched on an allocation. An entity
        must have an allocation assigned to it in the running settings
        or create_step will throw a LauncherError

        :param entity_name: name of the entity to create a step for
        :type entity_name: str
        :param run_settings: smartsim run settings for an entity
        :type run_settings: dict
        :param multi_prog: Create step with slurm --multi-prog, defaults to False
        :type multi_prog: bool, optional
        :raises LauncherError: if no allocation specified for the step
        :return: slurm job step
        :rtype: SlurmStep
        """
        try:
            if "alloc" not in run_settings:
                raise SSConfigError(f"User provided no allocation for {entity_name}")

            name = entity_name + "-" + str(np.base_repr(time.time_ns(), 36))
            step = SlurmStep(name, run_settings, multi_prog)
            return step
        except SSConfigError as e:
            raise LauncherError("Job step creation failed: " + str(e)) from None

    def get_step_status(self, step_id):
        """Get the status of a SlurmStep via the id of the step (e.g. 12345.0)

        :param step_id: id of the step in the form of xxxxxx.x
        :type step_id: str
        :return: status of the job step and returncode
        :rtype: SlurmStepInfo
        """
        step_id = str(step_id)
        sacct_out, _ = sacct(["--noheader", "-p", "-b", "-j", step_id])
        stat, returncode = parse_sacct(sacct_out, step_id)
        if self.task_manager.check_error(step_id):
            task_ret_code, out, err = self.task_manager.get_task_history(step_id)

            if stat == "NOTFOUND":
                returncode = task_ret_code
                stat = "Failed"
            status = SlurmStepInfo(stat, returncode, out, err)
        else:
            status = SlurmStepInfo(stat, returncode)
        return status

    def get_step_update(self, step_ids):
        """Get updates for a list of step ids

        :param step_ids: list of step ids
        :type step_ids: list
        :return: list of SlurmStepInfo instances
        :rtype: list
        """
        step_str = _create_step_id_str(step_ids)
        sacct_out, _ = sacct(["--noheader", "-p", "-b", "--jobs", step_str])
        # (status, returncode)
        stat_tuples = [parse_sacct(sacct_out, step_id) for step_id in step_ids]

        # create SlurmStepInfo objects to return
        updates = []
        for stat_tuple, step_id in zip(stat_tuples, step_ids):
            info = SlurmStepInfo(stat_tuple[0], stat_tuple[1])
            if self.task_manager.check_error(step_id):
                rc, out, err = self.task_manager.get_task_history(step_id)
                info.output = out
                info.error = err

                # command never made it to slurm, return Popen returncode
                if info.status == "NOTFOUND":
                    info.returncode = rc
                    info.status = "Failed"
            updates.append(info)
        return updates

    def get_step_nodes(self, step_id):
        """Return the compute nodes of a specific job or allocation

        This function returns the compute nodes of a specific job or allocation
        in a list with the duplicates removed.

        :param step_id: job step id or allocation id
        :type step_id: str
        :raises LauncherError: if allocation or job step cannot be
                               found
        :return: list of compute nodes the job was launched on
        :rtype: list of str
        """
        step_id = str(step_id)
        output, error = sstat([step_id, "-i", "-n", "-p", "-a"])
        if "error:" in error.split(" "):
            raise LauncherError("Could not find allocation for job: " + step_id)
        return parse_sstat_nodes(output)

    def run(self, step):
        """Run a job step

        This function runs a job step on an allocation through the
        slurm launcher. A constructed job step is required such that
        the argument translation from SmartSimEntity to SlurmStep has
        been completed and an allocation has been assigned to the step.

        :param step: Job step to be launched
        :type step: SlurmStep
        :raises LauncherError: If the allocation cannot be found or the
                               job step failed to launch.
        :return: job_step id
        :rtype: str
        """
        self.check_for_slurm()
        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        srun = step.build_cmd()
        task = execute_async_cmd(srun, cwd=step.cwd)
        step_id = self._get_slurm_step_id(step)
        self.task_manager.add_task(task, step_id)
        return step_id

    def stop(self, step_id):
        """Stop a job step

        :param step_id: id of the step to be stopped
        :type step_id: str
        :return: a SlurmStepInfo instance
        :rtype: SlurmStepInfo
        """
        self.check_for_slurm()
        scancel_rc, out, err = scancel([str(step_id)])
        self.task_manager.remove_task(step_id)
        if scancel_rc != 0:
            logger.warning(f"Unable to stop job {str(step_id)}")
        else:
            logger.info(f"Successfully stopped job {str(step_id)}")
        returncode, out, err = self.task_manager.get_task_history(step_id)
        info = SlurmStepInfo("Cancelled", returncode, out, err)
        return info

    def _get_slurm_step_id(self, step, interval=1, trials=5):
        """Get the step_id of a step
        This function uses the sacct command to find the step_id of
        a step that has been launched by this SlurmLauncher instance.
        Use the step name to find the corresponding step_id.  This
        function performs a number of trials in case the slurm launcher
        takes a while to populate the Sacct database

        TODO update this docstring
        :param step: step to find the id of
        :type step: SlurmStep
        :returns: if of the step
        :rtype: str
        """
        time.sleep(interval)
        step_id = "unassigned"
        while trials > 0:
            output, _ = sacct(
                ["--noheader", "-p", "--format=jobname,jobid", "--job=" + step.alloc_id]
            )
            step_id = parse_step_id_from_sacct(output, step.name)
            if step_id:
                break
            else:
                time.sleep(interval)
                trials -= 1
        if not step_id:
            raise LauncherError("Could not find id of launched job step")
        return step_id

    @staticmethod
    def check_for_slurm():
        """Check if slurm is available

        This function checks for slurm commands where the experiment
        is bring run

        :raises LauncherError: if no access to slurm
        """
        if not which("srun") and not which("salloc") and not which("sacct"):
            error = "User attempted Slurm methods without access to Slurm at the call site.\n"
            raise LauncherError(error)

    def __str__(self):
        return "slurm"


def _create_step_id_str(step_ids):
    step_str = ""
    for step_id in step_ids:
        step_str += str(step_id) + ","
    step_str = step_str.strip(",")
    return step_str
