from ...error import LauncherError, SSConfigError
from ...utils.helpers import expand_exe_path
from ..shell import execute_cmd


def sstat(args):
    """Calls sstat with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sstat
    """
    _sstat = _find_slurm_command("sstat")
    cmd = [_sstat] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def sacct(args):
    """Calls sacct with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sacct
    """
    _sacct = _find_slurm_command("sacct")
    cmd = [_sacct] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def salloc(args):
    """Calls slurm salloc with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of salloc
    """
    _salloc = _find_slurm_command("salloc")
    cmd = [_salloc] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def sinfo(args):
    """Calls slurm sinfo with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sinfo
    """
    _sinfo = _find_slurm_command("sinfo")
    cmd = [_sinfo] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def scancel(args):
    """Calls slurm scancel with args.

    returncode is also supplied in this function.

    :param args: list of command arguments
    :type args: list of str
    :return: output and error
    :rtype: str
    """
    _scancel = _find_slurm_command("scancel")
    cmd = [_scancel] + args
    returncode, out, error = execute_cmd(cmd)
    return returncode, out, error


def _find_slurm_command(cmd):
    try:
        full_cmd = expand_exe_path(cmd)
        return full_cmd
    except SSConfigError as e:
        raise LauncherError(f"Slurm Launcher could not find path of {cmd} command") from e
