import argparse
import importlib
import sys
import subprocess as sp
import typing as t

from smartsim._core._cli.utils import MenuItemConfig


def dynamic_execute(cmd: str) -> t.Callable[[argparse.Namespace, t.List[str]], int]:
    def process_execute(_args: argparse.Namespace, unparsed_args: t.List[str]) -> int:
        try:
            importlib.find_spec(cmd)
        except (ModuleNotFoundError, AttributeError):
            print(f"{cmd} plugin not found. Please ensure it is installed")
            return 1

        combined_cmd = [sys.executable, "-m", cmd] + unparsed_args
        with sp.Popen(combined_cmd, stdout=sp.PIPE, stderr=sp.PIPE) as process:
            stdout, _ = process.communicate()
            while process.returncode is None:
                stdout, _ = process.communicate()

            plugin_stdout = stdout.decode("utf-8")
            print(plugin_stdout)
            return process.returncode

    return process_execute


def dashboard() -> MenuItemConfig:
    return MenuItemConfig(
        "dashboard",
        "Start the SmartSim dashboard",
        dynamic_execute("smartdashboard.Experiment_Overview"),
        is_plugin=True,
    )

plugins = (dashboard, )
