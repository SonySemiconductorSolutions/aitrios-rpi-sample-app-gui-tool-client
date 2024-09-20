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

import ctypes
import os
import queue
import struct
import threading
import time
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np

from unify.models import MODEL_TYPE, Model

from ..device import Device
from ..frame import Frame

NETWORK_NAME_LEN = 64
MAX_NUM_TENSORS = 8
MAX_NUM_DIMENSIONS = 8


class OutputTensorInfo(ctypes.LittleEndianStructure):
    _fields_ = [
        ("tensor_data_num", ctypes.c_uint32),
        ("num_dimensions", ctypes.c_uint32),
        ("size", ctypes.c_uint16 * MAX_NUM_DIMENSIONS),
    ]


class CnnOutputTensorInfoExported(ctypes.LittleEndianStructure):
    _fields_ = [
        ("network_name", ctypes.c_char * NETWORK_NAME_LEN),
        ("num_tensors", ctypes.c_uint32),
        ("info", OutputTensorInfo * MAX_NUM_TENSORS),
    ]


class AiCamera(Device):

    def __init__(self, headless: Optional[bool] = False, timeout: Optional[int] = None):
        self.model = None
        self.picam2 = None

        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_thread = threading.Event()

        self.last_detections = None
        self.detection_times = deque(maxlen=30)
        self.last_detection_time = time.perf_counter()
        self.dps = 0

        super().__init__(headless=headless, timeout=timeout)

    # <------------ Entrypoints ------------>
    def deploy(self, model: Model):
        self.model = model

        if model.model_type != MODEL_TYPE.PACKAGED:
            raise ValueError("Deploy only supports packaged models")

        if not model.model_file.endswith(".rpk"):
            raise ValueError("Packeged model expects a .rpk file ending")

        imx500 = self.get_imx500_model(model.model_file)
        if model.preserve_aspect_ratio:
            imx500.set_inference_aspect_ratio(imx500.config["input_tensor_size"])

        # After deploy initiate Picamera2 (reads the symlink)
        self.picam2 = self.initiate_picamera2()
        self.picam2_start()

    def picam2_start(self):

        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888"}, controls={"FrameRate": 30, "CnnEnableInputTensor": False}, buffer_count=28
        )

        self.image = None

        def pre_callback(request):
            from picamera2 import MappedArray

            with MappedArray(request, "main") as m:
                self.image = m.array.copy()
                self.height, self.width, self.num_channels = np.shape(self.image)

        self.picam2.start(config, show_preview=False)
        self.picam2.pre_callback = pre_callback

    def get_output_shapes(self, metadata: dict) -> List[List[int]]:
        """
        Get the model output shapes if no output return empty list
        """
        output_tensor_info = metadata.get("CnnOutputTensorInfo")
        if not output_tensor_info:
            return []

        output_tensor_info = self.__get_output_tensor_info(output_tensor_info)["info"]
        return [o["size"] for o in output_tensor_info]

    def __get_output_tensor_info(self, tensor_info) -> dict:
        """
        Return the network string along with a list of output tensor parameters.
        """
        if type(tensor_info) not in [bytes, bytearray]:
            tensor_info = bytes(tensor_info)

        size = ctypes.sizeof(CnnOutputTensorInfoExported)
        if len(tensor_info) != size:
            raise ValueError(f"tensor info length {len(tensor_info)} does not match expected size {size}")

        # Create an instance of the struct and copy data into it
        parsed = CnnOutputTensorInfoExported()
        ctypes.memmove(ctypes.addressof(parsed), tensor_info, size)

        result = {
            "network_name": parsed.network_name.decode("utf-8").strip("\x00"),
            "num_tensors": parsed.num_tensors,
            "info": [
                {
                    "tensor_data_num": t.tensor_data_num,
                    "num_dimensions": t.num_dimensions,
                    "size": list(t.size)[0 : t.num_dimensions],
                }
                for t in parsed.info[0 : parsed.num_tensors]
            ],
        }

        return result

    def picam2_thread_function(self, queue, model):

        if self.model is None:
            self.picam2 = self.initiate_picamera2()
            self.picam2_start()

        while not self.stop_thread.is_set():
            try:
                metadata = self.picam2.capture_metadata()
                output_tensor = metadata.get("CnnOutputTensor")
                new_detection = False

                # Process output tensor
                if model is None:
                    detections = None
                elif output_tensor:

                    # reshape buffer to tensor shapes
                    np_output = np.fromiter(output_tensor, dtype=np.float32)
                    output_shapes = self.get_output_shapes(metadata)

                    offset = 0
                    outputs = []
                    for tensor_shape in output_shapes:
                        size = np.prod(tensor_shape)
                        outputs.append(np_output[offset : offset + size].reshape(tensor_shape, order="F"))
                        offset += size

                    # Post processing
                    detections = model.post_process(outputs)

                    new_detection = True
                    self.last_detections = detections
                    self.update_dps()
                elif self.last_detections is None:
                    # Missing output tensor in frame (no detection yet)
                    continue
                else:
                    # Missing output tensor in frame
                    detections = self.last_detections

                # Append frame to frame queue
                queue.put(
                    Frame(
                        timestamp=datetime.now().isoformat(),
                        image=self.image,
                        width=self.width,
                        height=self.height,
                        channels=self.num_channels,
                        detections=detections,
                        new_detection=new_detection,
                        fps=self.fps,
                        dps=self.dps,
                        color_format="BGR",
                    )
                )

            except KeyError:
                pass

        self.picam2.close()

    def update_dps(self):
        current_time = time.perf_counter()
        self.detection_times.append(current_time - self.last_detection_time)
        self.last_detection_time = current_time
        if len(self.detection_times) > 1:
            self.dps = len(self.detection_times) / sum(self.detection_times)

    # <------------ Stream ------------>
    def __enter__(self):
        self.stop_thread.clear()

        self.picam2_thread = threading.Thread(target=self.picam2_thread_function, args=(self.frame_queue, self.model))
        self.picam2_thread.start()

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_thread.set()
        self.picam2_thread.join()

    def __iter__(self):
        self.last_time = time.perf_counter()
        return self

    def __next__(self):
        self.check_timeout()
        self.update_fps()

        try:
            return self.frame_queue.get(timeout=120)
        except queue.Empty:
            raise StopIteration

    @staticmethod
    def initiate_picamera2():
        try:
            from picamera2 import Picamera2

            return Picamera2()
        except ImportError:
            raise ImportError(
                """
                picamera2 is not installed. Please install picamera2 to use the AiCamera device.\n\n
                For a raspberry pi with picamera2 installed. Enable in virtual env using:
                `python -m venv .venv --system-site-packages`\n
                """
            )

    @staticmethod
    def get_imx500_model(model_path):
        try:
            from picamera2.devices.imx500 import IMX500

            return IMX500(os.path.abspath(model_path))
        except ImportError:
            raise ImportError(
                """
                picamera2 is not installed. Please install picamera2 to use the AiCamera device.\n\n
                For a raspberry pi with picamera2 installed. Enable in virtual env using:
                `python -m venv .venv --system-site-packages`\n
                """
            )
