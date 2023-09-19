# BSD 2-Clause License
#
# Copyright (c) 2021-2023 Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import logging
from multiprocessing import RLock
import os
import pathlib
import signal
import typing as t


from datetime import datetime
from types import FrameType


from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, LoggingEventHandler
from watchdog.events import FileCreatedEvent, FileModifiedEvent
from smartsim._core.control.job import Job
from smartsim._core.control.jobmanager import JobManager
from smartsim._core.launcher.stepInfo import StepInfo
from smartsim._core.launcher.cobalt.cobaltLauncher import CobaltLauncher

from smartsim._core.launcher.launcher import Launcher
from smartsim._core.launcher.local.local import LocalLauncher
from smartsim._core.launcher.lsf.lsfLauncher import LSFLauncher
from smartsim._core.launcher.pbs.pbsLauncher import PBSLauncher
from smartsim._core.launcher.slurm.slurmLauncher import SlurmLauncher
from smartsim.experiment import Experiment
from smartsim.status import TERMINAL_STATUSES


logging.basicConfig(level=logging.INFO)

"""
Telemetry Monitor entrypoint
"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
_EventClass = t.Literal["start", "stop", "timestep"]
_ManifestKey = t.Literal["timestamp", "applications", "orchestrators", "ensembles"]
_JobKey = t.Tuple[str, str]


@dataclass
class PersistableEntity:
    entity_type: str
    name: str
    job_id: str
    step_id: str
    timestamp: int
    path: str

    @property
    def key(self) -> _JobKey:
        return (self.job_id, self.step_id)

    @property
    def is_orch(self) -> bool:
        return self.entity_type == "orchestrator"

    @property
    def is_managed(self) -> bool:
        return True if self.step_id else False


_FilterFn = t.Callable[[PersistableEntity], bool]


@dataclass
class Run:
    timestamp: int
    applications: t.List[PersistableEntity]
    orchestrators: t.List[PersistableEntity]
    ensembles: t.List[PersistableEntity]

    def flatten(
        self, filter_fn: t.Optional[_FilterFn] = None
    ) -> t.List[PersistableEntity]:
        """Flatten runs into a list of SmartSimEntity run events"""
        entities = self.applications + self.orchestrators + self.ensembles
        if filter_fn:
            entities = [entity for entity in entities if filter_fn(entity)]
        return entities


@dataclass
class ExperimentManifest:
    name: str
    path: pathlib.Path
    launcher: str
    runs: t.List[Run] = field(default_factory=list)


def hydrate_persistable(
    entity_type: str,
    persisted_entity: t.Dict[str, t.Any],
    timestamp: int,
    exp_dir: pathlib.Path,
) -> PersistableEntity:
    """Map entity data persisted in a manifest file to an object"""
    return PersistableEntity(
        entity_type=entity_type,
        name=persisted_entity["name"],
        job_id=persisted_entity.get("job_id", ""),
        step_id=persisted_entity.get("step_id", ""),
        timestamp=timestamp,
        path=str(exp_dir),
    )


def hydrate_persistables(
    entity_type: _ManifestKey,
    run: t.Dict[_ManifestKey, t.Any],
    exp_dir: pathlib.Path,
) -> t.List[PersistableEntity]:
    """Map a collection of entity data persisted in a manifest file to an object"""
    ts = run["timestamp"]
    return [
        hydrate_persistable(entity_type, item, ts, exp_dir) for item in run[entity_type]
    ]


def hydrate_runs(
    persisted_runs: t.List[t.Dict[_ManifestKey, t.Any]], exp_dir: pathlib.Path
) -> t.List[Run]:
    """Map run data persisted in a manifest file to an object"""
    runs = [
        Run(
            timestamp=instance["timestamp"],
            applications=hydrate_persistables("applications", instance, exp_dir),
            orchestrators=hydrate_persistables("orchestrators", instance, exp_dir),
            ensembles=hydrate_persistables("ensembles", instance, exp_dir),
        )
        for instance in persisted_runs
    ]
    return runs


def load_manifest(file_path: str) -> ExperimentManifest:
    """Load a persisted manifest and return the content"""
    source = pathlib.Path(file_path)
    source = source.resolve()

    text = source.read_text(encoding="utf-8")
    manifest_dict = json.loads(text)

    exp_dir = pathlib.Path(manifest_dict["experiment"]["path"])

    manifest = ExperimentManifest(
        name=manifest_dict["experiment"]["name"],
        path=exp_dir,
        launcher=manifest_dict["experiment"]["launcher"],
        runs=hydrate_runs(manifest_dict["runs"], exp_dir),
    )
    return manifest


def track_event(
    timestamp: int,
    entity: PersistableEntity,
    action: _EventClass,
    exp_dir: pathlib.Path,
    logger: logging.Logger,
    detail: str = "",
) -> None:
    """
    Persist a tracking event for an entity
    """
    job_id = entity.job_id or ""
    step_id = entity.step_id or ""
    entity_type = entity.entity_type or "missing_entity_type"
    logger.info(
        f"mocked tracking `{entity_type}.{action}` event w/jid: {job_id}, "
        f"tid: {step_id}, ts: {timestamp}"
    )

    name: str = entity.name or "entity-name-not-found"
    tgt_path = exp_dir / "manifest" / entity_type / name / f"{action}.json"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        entity_dict = {**entity.__dict__}
        entity_dict.pop("path", None)
        entity_dict["detail"] = detail
        tgt_path.write_text(json.dumps(entity_dict))
    except Exception as ex:
        print(ex)


def track_completed(job: Job, logger: logging.Logger) -> None:
    """Persists telemetry event for the end of job"""
    timestamp = datetime.timestamp(datetime.now())
    inactive_entity = job.entity
    detail = job.status
    exp_dir = pathlib.Path(job.entity.path)

    track_event(timestamp, inactive_entity, "stop", exp_dir, logger, detail=detail)


def track_started(job: Job, logger: logging.Logger) -> None:
    """Persists telemetry event for the start of job"""
    timestamp = datetime.timestamp(datetime.now())
    inactive_entity = job.entity
    exp_dir = pathlib.Path(job.entity.path)

    track_event(timestamp, inactive_entity, "start", exp_dir, logger)


def track_timestep(job: Job, logger: logging.Logger) -> None:
    """Persists telemetry event for a timestep"""
    timestamp = datetime.timestamp(datetime.now())
    inactive_entity = job.entity
    timestamp_suffix = str(int(timestamp))  # drop floating point part before stringify
    exp_dir = pathlib.Path(job.entity.path)

    track_event(timestamp, inactive_entity, f"step_{timestamp_suffix}", exp_dir, logger)


class ManifestEventHandler(PatternMatchingEventHandler):
    """The ManifestEventHandler monitors an experiment for changes and updates 
    a telemetry datastore as needed. 
    
    It contains event handlers that are triggered by changes to a runtime experiment
    manifest. The runtime manifest differs from a standard manifest. A runtime manifest
    may contain multiple experiment executions in a `runs` collection.
    
    It also contains a long-polling loop that checks experiment entities for updates
    at each timestep."""
    def __init__(
        self,
        pattern: str,
        logger: logging.Logger,
        ignore_patterns: t.Any = None,
        ignore_directories: bool = True,
        case_sensitive: bool = False,
    ) -> None:
        super().__init__(
            [pattern], ignore_patterns, ignore_directories, case_sensitive
        )  # type: ignore
        self._logger = logger
        self._tracked_runs: t.Dict[int, Run] = {}
        self._tracked_jobs: t.Dict[_JobKey, PersistableEntity] = {}
        self._completed_jobs: t.Dict[_JobKey, PersistableEntity] = {}
        self._launcher_type: str = ""
        self._launcher: t.Optional[Launcher] = None
        self._jm: JobManager = JobManager(RLock())

    def init_launcher(self, launcher: str) -> None:
        """Initialize the controller with a specific type of launcher.
        SmartSim currently supports slurm, pbs(pro), cobalt, lsf,
        and local launching

        :param launcher: which launcher to initialize
        :type launcher: str
        :raises SSUnsupportedError: if a string is passed that is not
                                    a supported launcher
        :raises TypeError: if no launcher argument is provided.
        """
        launcher_map: t.Dict[str, t.Type[Launcher]] = {
            "slurm": SlurmLauncher,
            "pbs": PBSLauncher,
            "cobalt": CobaltLauncher,
            "lsf": LSFLauncher,
            "local": LocalLauncher,
        }

        if launcher is not None:
            launcher = launcher.lower()
            if launcher in launcher_map:
                return launcher_map[launcher]()
            else:
                raise ValueError("Launcher type not supported: " + launcher)
        else:
            raise TypeError("Must provide a 'launcher' argument")

    def set_launcher(self, launcher_type: str) -> None:
        """Set the launcher for the experiment"""
        if launcher_type != self._launcher_type:
            self._launcher_type = launcher_type
            self._launcher = self.init_launcher(
                launcher_type
            )
            self._jm.set_launcher(self._launcher)
            self._jm.add_job_onstart_callback(track_started)
            self._jm.add_job_onstop_callback(track_completed)
            self._jm.add_job_onstep_callback(track_timestep)
            self._jm.start()

    @property
    def launcher(self) -> Launcher:
        """Return a launcher appropriate for the experiment"""
        if not self._launcher:
            self._launcher = LocalLauncher()

        # if not self._launcher:
        #     raise ValueError("Launcher failed to instantiate properly")

        return self._launcher

    def _process_manifest(self, manifest_path: str, logger: logging.Logger) -> None:
        """Load the runtime manifest for the experiment and add the entities
        to the collections of items being tracked for updates"""
        manifest = load_manifest(manifest_path)
        self.set_launcher(manifest.launcher)

        runs = [run for run in manifest.runs if run.timestamp not in self._tracked_runs]

        # Find exp root assuming event path `{exp_root}/manifest/manifest.json`
        exp_dir = pathlib.Path(manifest_path).parent.parent

        for run in runs:
            for entity in run.flatten(
                filter_fn=lambda e: e.key not in self._tracked_jobs
            ):
                entity.path = str(exp_dir)

                self._tracked_jobs[entity.key] = entity
                track_event(run.timestamp, entity, "start", exp_dir, logger)

                self._jm.add_telemetry_job(
                    entity.name,
                    entity.job_id,
                    entity,
                    is_task=entity.is_managed,
                    is_orch=entity.is_orch,
                )
                self._jm._launcher.step_mapping.add(
                    entity.name, entity.step_id, entity.step_id, entity.is_managed
                )
            self._tracked_runs[run.timestamp] = run

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Event handler for when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`DirModifiedEvent` or :class:`FileModifiedEvent`
        """
        super().on_modified(event)  # type: ignore
        self._process_manifest(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Event handler for when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """
        super().on_created(event)  # type: ignore
        self._process_manifest(event.src_path)

    def _to_completed(
        self,
        timestamp: int,
        entity: PersistableEntity,
        exp_dir: pathlib.Path,
        step_info: StepInfo,
    ) -> None:
        """Move a monitored entity from the active to completed collection to
        stop monitoring for updates during timesteps."""
        inactive_entity = self._tracked_jobs.pop(entity.key)
        if entity.key not in self._completed_jobs:
            self._completed_jobs[entity.key] = inactive_entity
        
        job = self._jm[entity.name]
        self._jm.move_to_completed(job)

        detail = f"status: {step_info.status}, error: {step_info.error}"
        track_event(timestamp, entity, "stop", exp_dir, logger, detail=detail)

    def on_timestep(self, exp_dir: pathlib.Path) -> None:
        """Called at polling frequency to request status updates on monitored entities"""
        launcher = self.launcher
        entity_map = self._tracked_jobs

        names = {entity.name: entity for entity in entity_map.values()}
        timestamp = datetime.timestamp(datetime.now())

        if launcher and names:
            step_updates = launcher.get_step_update(list(names.keys()))

            for step_name, step_info in step_updates:
                if step_info.status in TERMINAL_STATUSES:
                    completed_entity = names[step_name]
                    self._to_completed(timestamp, completed_entity, exp_dir, step_info)


async def main(
    frequency: t.Union[int, float], experiment_dir: pathlib.Path, logger: logging.Logger
) -> None:
    """Setup the monitoring entities and start the timer-based loop that
    will poll for telemetry data"""
    logger.info(
        f"Executing telemetry monitor with frequency: {frequency}"
        f", on target directory: {experiment_dir}"
    )

    manifest_path = experiment_dir / "manifest" / "manifest.json"
    manifest_dir = str(experiment_dir / "manifest")
    logger.debug(f"Monitoring manifest changes at: {manifest_dir}")

    log_handler = LoggingEventHandler(logger)  # type: ignore
    action_handler = ManifestEventHandler("manifest.json", logger)

    observer = Observer()

    try:
        if manifest_path.exists():
            action_handler._process_manifest(str(manifest_path))

        observer.schedule(log_handler, manifest_dir)  # type: ignore
        observer.schedule(action_handler, manifest_dir)  # type: ignore
        observer.start()  # type: ignore

        while observer.is_alive():
            logger.debug(f"Telemetry timestep: {datetime.timestamp(datetime.now())}")
            action_handler.on_timestep(experiment_dir)
            await asyncio.sleep(frequency)
    except Exception as ex:
        logger.error(ex)
    finally:
        observer.join()
        observer.stop()  # type: ignore


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Event handler for os signals to cancel all tasks in the process"""
    if not signo:
        logger = logging.getLogger()
        logger.warning("Received signal with no signo")

    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()


def register_signal_handlers() -> None:
    """Register a signal handling function for all termination events"""
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def build_arg_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments"""
    parser = argparse.ArgumentParser(description="SmartSim Telemetry Monitor")
    parser.add_argument(
        "-f", type=str, help="Frequency of telemetry updates", default=5
    )
    parser.add_argument(
        "-d",
        type=str,
        help="Experiment root directory",
        # required=True,
        default="/lus/cls01029/mcbridch/ss/smartsim",
    )
    return parser


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = build_arg_parser()
    args = parser.parse_args()
    logger = logging.getLogger()

    # Must register cleanup before the main loop is running
    register_signal_handlers()

    try:        
        loop_fn = main(int(args.f), pathlib.Path(args.d), logger)
        asyncio.run(loop_fn)
    except asyncio.CancelledError:
        logger.exception("Shutting down telemetry monitor due to CancelledError")