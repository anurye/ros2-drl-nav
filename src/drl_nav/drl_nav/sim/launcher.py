# Copyright 2025 Ahmed Yesuf Nurye.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Start and stop a ROS 2 launch file as a background process."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Dict, Optional, Union


class SimulationLauncher:
    """Run ``ros2 launch`` in its own process group so it can be stopped cleanly."""

    def __init__(
        self,
        package: str = "drl_nav",
        launch_file: str = "simulation.launch.py",
        launch_args: Optional[Dict[str, Union[str, bool, float, int]]] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.package = package
        self.launch_file = launch_file
        self.launch_args = dict(launch_args or {})
        self.log_path = Path(log_path) if log_path else None
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None

    @property
    def command(self) -> list:
        args = [
            f"{key}:={_to_launch_value(value)}"
            for key, value in self.launch_args.items()
        ]
        return ["ros2", "launch", self.package, self.launch_file, *args]

    def start(self) -> "SimulationLauncher":
        if shutil.which("ros2") is None:
            raise RuntimeError(
                "`ros2` not found on PATH; source your ROS 2 workspace first."
            )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w")
        self._process = subprocess.Popen(
            self.command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT if self._log_file else None,
            start_new_session=True,
        )
        return self

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self, timeout_sec: float = 20.0) -> None:
        """Send SIGINT, then SIGTERM, then SIGKILL to the process group."""
        if self._process is None:
            return
        if self._process.poll() is None:
            pgid = os.getpgid(self._process.pid)
            for sig, wait in ((signal.SIGINT, timeout_sec), (signal.SIGTERM, 5.0)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                if _wait(self._process, wait):
                    break
            else:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._process = None

    def __enter__(self) -> "SimulationLauncher":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _to_launch_value(value: Union[str, bool, float, int]) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _wait(process: subprocess.Popen, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.1)
    return process.poll() is not None
