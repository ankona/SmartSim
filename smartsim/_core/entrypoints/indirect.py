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
import os
import pathlib
import psutil
import signal
import sys
from smartsim._core.control.job import JobEntity
import smartsim._core.entrypoints.telemetrymonitor as tm
import time
import typing as t

from types import FrameType

from smartsim.log import get_logger
from smartsim._core.entrypoints.telemetrymonitor import track_event
from smartsim._core.utils.helpers import get_ts, decode_cmd


STEP_PID: t.Optional[int] = None
logger = get_logger(__name__)

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]


def _get_target(run: tm.Run,
                etype: str,
                step_name: str) -> t.Optional[JobEntity]:
    list_map = {
        "model": run.models,
        "ensemble": run.ensembles,
        "orchestrator": run.orchestrators,
    }

    for item in  list_map[etype]:
        # note: matching name is problematic but the job manager is
        # tracking the proxy task instead of the actual task
        # if item.job_id == str(STEP_PID):
        #     return item
        if step_name.startswith(f"{item.name}-"):
            return item
    return None


def _track_target_start(mani_path: pathlib.Path,
                        step_name: str,
                        start_ts: int,
                        etype: str) -> t.Optional[JobEntity]:
    if not mani_path.exists():
        return None

    runtime_manifest = tm.load_manifest(str(mani_path))
    if not runtime_manifest:
        return None

    for run in runtime_manifest.runs:
        target = _get_target(run, etype, step_name)
        if target is not None:
            track_event(
                start_ts,
                target.job_id,
                target.step_id,
                target.type,
                "start",
                pathlib.Path(target.status_dir),
                logger,
            )
            return target
    return None


def main(
    cmd: str,
    etype: str,
    output_path: str,
    error_path: str,
    cwd: str,
    exp_dir: str,
    step_name: str,
) -> int:
    """Execute the step command and emit tracking events"""
    global STEP_PID  # pylint: disable=global-statement

    exp_path = pathlib.Path(exp_dir)
    if not exp_path.exists():
        raise ValueError(f"The experiment directory does not exist: {exp_dir}")

    if not cmd.strip():
        raise ValueError("Invalid cmd supplied")

    cleaned_cmd = decode_cmd(cmd)
    ret_code: int = 1
    logger.debug("Indirect step starting")

    with open(output_path, "w+", encoding="utf-8") as ofp, open(
        error_path, "w+", encoding="utf-8"
    ) as efp:
        try:
            process = psutil.Popen(
                cleaned_cmd,
                cwd=cwd,
                stdout=ofp.fileno(),
                stderr=efp.fileno(),
                close_fds=True,
            )
            STEP_PID = process.pid
            logger.info(f"Indirect step step {STEP_PID} started")
            start_ts = get_ts()

        except Exception:
            logger.error("Failed to create process", exc_info=True)
            cleanup()
            return 1

        mani_path = pathlib.Path(exp_dir) / ".smartsim/telemetry" / "manifest.json"
        target: t.Optional[tm.JobEntity] = None

        try:
            while all((process.is_running(), STEP_PID > 0)):
                if target is None:
                    target = _track_target_start(mani_path, step_name, start_ts, etype)

                if result := process.poll():
                    ret_code = result
                    break
                time.sleep(1)
        except Exception:
            logger.error("Failed to execute process", exc_info=True)
        finally:
            logger.info(f"Indirect step {STEP_PID} complete")

            if target is None:
                target = _track_target_start(mani_path, step_name, start_ts, etype)

            if target is not None:
                msg = f"process {target.step_id} finished with return code: {ret_code}"
                track_event(
                    get_ts(),
                    target.job_id,
                    target.step_id,
                    target.type,
                    "stop",
                    pathlib.Path(target.status_dir),
                    logger,
                    detail=msg,
                    return_code=ret_code,
                )
            cleanup()

    return ret_code


def cleanup() -> None:
    """Perform cleanup required for clean termination"""
    global STEP_PID  # pylint: disable=global-statement
    if STEP_PID is not None and STEP_PID < 1:
        return

    try:
        # attempt to stop the subprocess performing step-execution
        process = psutil.Process(STEP_PID)
        process.terminate()

    except psutil.NoSuchProcess:
        # swallow exception to avoid overwriting outputs from cmd
        ...

    except OSError as ex:
        logger.warning(f"Failed to clean up step executor gracefully: {ex}")
    finally:
        STEP_PID = 0


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Helper function to ensure clean process termination"""
    if not signo:
        logger.warning("Received signal with no signo")

    cleanup()


def register_signal_handlers() -> None:
    """Register a signal handling function for all termination events"""
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def get_parser() -> argparse.ArgumentParser:
    # NOTE: plus prefix avoids passing param incorrectly to python interpreter,
    # e.g. `python -m smartsim._core.entrypoints.indirect -c ... -t ...`
    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Step Executor"
    )
    parser.add_argument("+c", type=str, help="The command to execute", required=True)
    parser.add_argument(
        "+t", type=str, help="The type of entity related to the step", required=True
    )
    parser.add_argument(
        "+n", type=str, help="The step name being executed", required=True
    )
    parser.add_argument(
        "+d", type=str, help="The experiment root directory", required=True
    )
    parser.add_argument(
        "+w", type=str, help="The working directory of the executable", required=True
    )
    parser.add_argument("+o", type=str, help="Output file", required=True)
    parser.add_argument("+e", type=str, help="Erorr output file", required=True)
    return parser


if __name__ == "__main__":
    arg_parser = get_parser()
    os.environ["PYTHONUNBUFFERED"] = "1"

    try:
        parsed_args = arg_parser.parse_args()
        logger.debug("Starting indirect step execution")

        # make sure to register the cleanup before the start the process
        # so our signaller will be able to stop the database process.
        register_signal_handlers()

        rc = main(
            cmd=parsed_args.c,
            etype=parsed_args.t,
            output_path=parsed_args.o,
            error_path=parsed_args.e,
            cwd=parsed_args.w,
            exp_dir=parsed_args.d,
            step_name=parsed_args.n,
        )
        sys.exit(rc)

    # gracefully exit the processes in the distributed application that
    # we do not want to have start a colocated process. Only one process
    # per node should be running.
    except Exception as e:
        logger.exception(f"An unexpected error caused step execution to fail: {e}")
        sys.exit(1)
