#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from ..models import Model
from .frame import Frame


class Device(ABC):
    """
    Abstract base class for devices.

    A device in Unify has to conform to the following interface:
    1. Deploy a model to the device. Example usage:
    ```
    model = Model()
    device.deploy(model, *args)
    ```

    2. Enter and iterate over the frames in the device stream. Example usage:
    ```
    with device as stream:
        for frame in stream:
            ...
    ```
    """

    def __init__(self, headless: Optional[bool] = False, timeout: Optional[int] = None) -> None:

        self.headless = headless
        self.timeout = timeout

        self.frame_times = deque(maxlen=30)
        self.last_time = time.perf_counter()
        self.fps = 0

    @abstractmethod
    def deploy(self, model: Model, *args):
        """
        Abstract method to deploy a model to the device.

        Args:
            model (Model): The model to be deployed.
            *args: Additional arguments for the deployment.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Abstract method to enter a device stream.
        Assumes to set the start time for the device stream to check time-out.
        """
        self.start_time = time.perf_counter()
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Abstract method to exit the device stream.
        This method should handle cleaning up or closing the device,
        and possibly handling exceptions that occurred within the 'with' block.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Abstract method to iterate over the frames in the device stream.
        """
        pass

    @abstractmethod
    def __next__(self) -> Frame:
        """
        Abstract method to get the next frame in the device stream.

        Returns:
            Frame: The next frame in the device stream.
        """
        self.update_fps()
        self.check_timeout()
        pass

    def update_fps(self):
        """
        Utility method for updating the frames per second (FPS) value based on the time elapsed between frames.
        """
        current_time = time.perf_counter()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / sum(self.frame_times)

    def check_timeout(self):
        """
        Utility method for checking if the specified timeout if it has been set.
        Stops the stream iterator if the timeout has been exceeded.
        """
        elapsed_time = time.perf_counter() - self.start_time
        if self.timeout is not None and elapsed_time > self.timeout:
            self.__exit__(None, None, None)
            raise StopIteration
