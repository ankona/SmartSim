# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
import logging
import os
import sys
import typing as t

import coloredlogs

# constants
DEFAULT_DATE_FORMAT: t.Final[str] = "%H:%M:%S"
DEFAULT_LOG_FORMAT: t.Final[
    str
] = "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"

# configure colored loggs
coloredlogs.DEFAULT_DATE_FORMAT = DEFAULT_DATE_FORMAT
coloredlogs.DEFAULT_LOG_FORMAT = DEFAULT_LOG_FORMAT


def _get_log_level() -> str:
    """Get the logging level based on environment variable
       SMARTSIM_LOG_LEVEL.  If not set, default to info.

       Logging levels
         - quiet: Just shows errors and warnings
         - info: Show basic information and errors (default)
         - debug: Shows info, errors and user debug information
         - developer: Shows everything happening during execution
                      extremely verbose logging.

    :return: Log level for coloredlogs
    :rtype: str
    """
    log_level = os.environ.get("SMARTSIM_LOG_LEVEL", "info").lower()
    if log_level == "quiet":
        return "warning"
    if log_level == "info":
        return "info"
    if log_level == "debug":
        return "debug"
    # extremely verbose logging used internally
    if log_level == "developer":
        return "debug"
    return "info"


def get_logger(
    name: str, log_level: t.Optional[str] = None, fmt: t.Optional[str] = None
) -> logging.Logger:
    """Return a logger instance

    levels:
        - quiet
        - info
        - debug
        - developer

    examples:
        # returns a logger with the name of the module
        logger = get_logger(__name__)

        logger.info("This is a message")
        logger.debug("This is a debug message")
        logger.error("This is an error message")
        logger.warning("This is a warning message")

    :param name: the name of the desired logger
    :type name: str
    :param log_level: what level to set the logger to
    :type log_level: str
    :param fmt: the format of the log messages
    :type fmt: str
    :returns: logger instance
    :rtype: logging.Logger
    """
    # if name is None, then logger is the root logger
    # if not root logger, get the name of file without prefix.
    user_log_level = _get_log_level()
    if user_log_level != "developer":
        name = "SmartSim"

    logger = logging.getLogger(name)
    if log_level:
        logger.setLevel(log_level)
    else:
        log_level = user_log_level
    coloredlogs.install(level=log_level, logger=logger, fmt=fmt, stream=sys.stdout)

    return logger


def add_exp_loggers(
    exp_path: str, logger: logging.Logger, fmt: t.Optional[str] = None
) -> None:
    """Add FileHandlers to a logger instance for producing logs
    in an experiment directory"""
    logfile_enabled = os.environ.get("SMARTSIM_LOGFILE_ENABLED", "1")

    if int(logfile_enabled) > 0 and logger.name.lower().startswith("smartsim"):
        out_path = os.path.join(exp_path, "smartsim.out")
        err_path = os.path.join(exp_path, "smartsim.err")

        # log errors below warning to <outfile>.out & those above to <outfile>.err
        log_to_file(out_path, "INFO", logger, fmt, LevelFilter(maximum_level="INFO"))
        log_to_file(err_path, "WARN", logger, fmt)


class LevelFilter(logging.Filter):
    """A filter that passes all records below a desired level"""

    def __init__(self, maximum_level: str = "INFO"):
        """Create a high-pass log filter allowing messages below a specific log level

        :param maximum_level: The maximum log level to be passed by the filter
        :type maximum_level: str
        """
        super().__init__()
        self.max = maximum_level

    def filter(self, record: logging.LogRecord) -> bool:
        # If a string representation of the level is passed in,
        # the corresponding numeric value is returned.
        if record.levelno <= logging.getLevelName(self.max):
            return True
        return False


def log_to_file(
    filename: str,
    log_level: str = "warn",
    logger: t.Optional[logging.Logger] = None,
    fmt: t.Optional[str] = None,
    log_filter: t.Optional[logging.Filter] = None,
) -> None:
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param filename: the name of the desired log file.
    :type filename: str

    :param log_level: as defined in get_logger.  Can be specified
                      to allow the file to store more or less verbose
                      logging information.
    :type log_level: int | str
    """
    if logger is None:
        logger = logging.getLogger("SmartSim")

    handler = logging.FileHandler(filename, mode="a+", encoding="utf-8")

    if log_filter:
        handler.addFilter(log_filter)

    fmt = fmt or DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(fmt=fmt, datefmt=DEFAULT_DATE_FORMAT)
    handler.setFormatter(formatter)
    handler.setLevel(log_level.upper())

    logger.addHandler(handler)
