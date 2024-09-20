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

import sys
from typing import Optional, Union

import cv2
import numpy as np

from ..models import Classifications, Detections, Poses, Segments


class Frame:
    """
    Represents a frame in a device stream.

    Attributes:
        timestamp (str): The timestamp of the frame.
        image (np.ndarray): The image data of the frame.
        width (int): The width of the frame.
        height (int): The height of the frame.
        channels (int): The number of channels in the frame.
        detections (Union[Classifications, Detections, Poses, Segments]): The detections in the frame.
        new_detection (bool): Flag if the provided detections are updated or an old copy.
        fps (float): The frames per second of the video stream.
        dps (float): The detections per second in the video stream.
        color_format (str, optional): The color format of the frame. Defaults to "RGB".
    """

    def __init__(
        self,
        timestamp: str,
        image: np.ndarray,
        width: int,
        height: int,
        channels: int,
        detections: Union[Classifications, Detections, Poses, Segments],
        new_detection: bool,
        fps: float,
        dps: float,
        color_format: Optional[str] = "RGB",
    ):
        self.timestamp = timestamp
        self._image = image
        self.width = width
        self.height = height
        self.channels = channels
        self._detections = detections
        self.new_detection = new_detection
        self.fps = fps
        self.dps = dps
        self.color_format = color_format

    @property
    def image(self):
        """
        Get the image data of the frame.

        Returns:
            np.ndarray: The image data of the frame.

        Raises:
            ValueError: When running headless and the image is not available.
        """
        if self._image is not None:
            return self._image
        else:
            raise ValueError("Running headless: `frame.image` unavailable.\n")

    @image.setter
    def image(self, value):
        self._image = value

    @property
    def detections(self) -> Union[Classifications, Detections, Poses, Segments]:
        """
        Get the detections in the frame.

        Returns:
            Union[Classifications, Detections, Poses, Segments]: The detections in the frame.

        Raises:
            ValueError: If no model is running.
        """
        if self._detections is not None:
            return self._detections
        else:
            raise ValueError("No model is running: `frame.detections` unavailable.\n")

    def display(self):
        """
        Display the frame according to the specified color format.
        """
        cv2.namedWindow("Application", cv2.WINDOW_NORMAL)
        cv2.imshow(
            "Application",
            cv2.putText(
                cv2.putText(
                    (cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR) if self.color_format == "RGB" else self.image),
                    f"FPS: {self.fps:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                ),
                f"DPS: {self.dps:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            ),
        )

        # 'ESC' key or window is closed manually
        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Application", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            sys.exit()
