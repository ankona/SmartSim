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

import os
import stat
# import typing as t

from ....log import get_logger
# from ....settings import CobaltBatchSettings
from .step import BatchStepBase  # , Step

logger = get_logger(__name__)


class CobaltBatchStep(BatchStepBase):

    def _write_script(self) -> str:
        """Write the batch script

        :return: batch script path after writing
        :rtype: str
        """
        batch_script = self.get_step_file(ending=".sh")
        cobalt_debug = self.get_step_file(ending=".cobalt-debug")
        output, error = self.get_output_files()
        with open(batch_script, "w", encoding="utf-8") as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write(f"#COBALT -o {output}\n")
            script_file.write(f"#COBALT -e {error}\n")
            script_file.write(f"#COBALT --cwd {self.cwd}\n")
            script_file.write(f"#COBALT --jobname {self.name}\n")
            script_file.write(f"#COBALT --debuglog {cobalt_debug}\n")

            # add additional sbatch options
            for opt in self.batch_settings.format_batch_args():
                script_file.write(f"#COBALT {opt}\n")

            for cmd in self.batch_settings.preamble:
                script_file.write(f"{cmd}\n")

            for i, step_cmd in enumerate(self.step_cmds):
                script_file.write("\n")
                script_file.write(f"{' '.join((step_cmd))} &\n")
                if i == len(self.step_cmds) - 1:
                    script_file.write("\n")
                    script_file.write("wait\n")
        os.chmod(batch_script, stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR)
        return batch_script
