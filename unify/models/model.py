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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from .results import Classifications, Detections, Poses, Segments


@dataclass
class MODEL_TYPE:
    KERAS = "keras"
    ONNX = "onnx"
    TFLITE = "tflite"
    CONVERTED = "converted"
    PACKAGED = "packaged"


@dataclass
class COLOR_FORMAT:
    RGB = "RGB"
    BGR = "BGR"


class Model(ABC):
    """
    Abstract base class for models.

    Args:
        model_file (Path): The path to the model file.
        model_type (MODEL_TYPE): The type of the model.
        color_format (COLOR_FORMAT, optional): The color format of the model. Defaults to COLOR_FORMAT.RGB.
        preserve_aspect_ratio (bool, optional): Setting the sensor whether or not to preserve aspect ratio
                                                of the input tensor. Defaults to True.
    """

    def __init__(
        self,
        model_file: Path = None,
        model_type: MODEL_TYPE = None,
        color_format: COLOR_FORMAT = COLOR_FORMAT.RGB,
        preserve_aspect_ratio: bool = True,
    ):
        self.model_file = os.path.abspath(model_file) if model_file else None
        self.model_type = model_type
        self.color_format = color_format
        self.preserve_aspect_ratio = preserve_aspect_ratio

    @abstractmethod
    def post_process(self, output_tensors: List[np.ndarray]) -> Union[Classifications, Detections, Poses, Segments]:
        """
        Perform post-processing on the tensor data and tensor layout.

        Args:
            output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

        Returns:
            Union[Classifications, Detections, Poses, Segments]: The post-processed detections.
        """
        pass
