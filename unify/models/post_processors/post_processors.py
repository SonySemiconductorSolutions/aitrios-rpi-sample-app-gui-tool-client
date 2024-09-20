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
import platform
from typing import List

import numpy as np

from ..results import Classifications, Detections, Poses, Segments
from .posenet_decoder.decoder_interfaces import PosenetOutputDataType


def pp_cls(output_tensors: List[np.ndarray]) -> Classifications:
    """
    Performs post-processing on a Classification result tensor.

    Args:
        output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

    Returns:
        Classifications: The post-processed classification detections.
    """

    # Creating sorted scores and indices arrays
    # Sorting the indices based on scores, negative for descending order
    sorted_indices = np.argsort(-output_tensors[0])

    return Classifications(confidence=output_tensors[0][sorted_indices], class_id=sorted_indices)


def pp_cls_softmax(output_tensors: List[np.ndarray]) -> Classifications:
    """
    Performs post-processing on a Classification result tensor that requires an additional softmax.

    Args:
        output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

    Returns:
        Classifications: The post-processed classification detections with softmax-applied confidence scores.
    """

    # Softmax
    y = np.exp(output_tensors[0] - np.expand_dims(np.max(output_tensors[0], axis=-1), axis=-1))
    np_output = y / np.expand_dims(np.sum(y, axis=-1), axis=-1)

    # Creating sorted scores and indices arrays
    # Sorting the indices based on scores, negative for descending order
    sorted_indices = np.argsort(-np_output)

    return Classifications(confidence=np_output[sorted_indices], class_id=sorted_indices)


def pp_od_bcsn(output_tensors: List[np.ndarray]) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor.
    Output tensor order: Boxes - Classes - Scores - Number

    Args:
        output_tensor (np.ndarray): Resulting output tensor to be processed.

    Returns:
        Detections: The post-processed object detection detections.
    """

    n_detections = int(output_tensors[3][0])

    return Detections(
        bbox=output_tensors[0][:n_detections][:, [1, 0, 3, 2]],
        class_id=np.array(output_tensors[1][:n_detections], dtype=np.uint16),
        confidence=output_tensors[2][:n_detections],
    )


def pp_od_bscn(output_tensors: List[np.ndarray]) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor.
    Output tensor order: Boxes - Scores - Classes - Number

    Args:
        output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

    Returns:
        Detections: The post-processed object detection detections.
    """

    n_detections = int(output_tensors[3][0])

    return Detections(
        bbox=output_tensors[0][:n_detections][:, [1, 0, 3, 2]],
        class_id=np.array(output_tensors[2][:n_detections], dtype=np.uint16),
        confidence=output_tensors[1][:n_detections],
    )


def pp_od_efficientdet_lite0(output_tensors: List[np.ndarray]) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor specifically for EfficientDet-Lite0.

    Args:
        output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

    Returns:
        Detections: The post-processed object detection detections, with bounding box coordinates normalized to a 320x320 scale.
    """

    detections = pp_od_bscn(output_tensors)
    detections.bbox /= 320
    return detections


arch = platform.machine()
if arch == 'x86_64':
    lib_name = "libposenet_amd64.so"
elif arch == 'aarch64':
    lib_name = "libposenet_arm64.so"
else:
    raise RuntimeError(f"Unsupported architecture: {arch}")

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), "posenet_decoder", lib_name))
lib.decode_poses.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(PosenetOutputDataType),
]
lib.decode_poses.restype = None


def pp_posenet(output_tensors: List[np.ndarray]) -> Poses:
    """
    Performs post-processing on a Posenet result tensor.

    Args:
        output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

    Returns:
        Poses: The post-processed pose estimation results.

    The output tensor is post processed by the posenet decoder handled with a binding to C++.
    The interface of this function: PosenetOutputDataType is populated with the decoded pose data, including:
        - Number of detections (n_detections).
        - Pose scores (pose_scores).
        - Keypoints for each detected pose (pose_keypoints).
        - Scores for each keypoint in the detected poses (pose_keypoint_scores).
    """
    result = PosenetOutputDataType()

    lib.decode_poses(
        output_tensors[0].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_tensors[1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_tensors[2].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(result),
    )

    # Normalize keypoints between 0 and 1
    input_tensor_size = (481, 353)
    keypoints = np.array(
        [np.ctypeslib.as_array(result.pose_keypoints[i].keypoints) for i in range(result.n_detections)]
    )
    if result.n_detections > 0:
        keypoints[:, ::2] /= input_tensor_size[0]
        keypoints[:, 1::2] /= input_tensor_size[1]

    return Poses(
        n_detections=result.n_detections,
        scores=np.ctypeslib.as_array(result.pose_scores),
        keypoints=keypoints,
        keypoint_scores=np.array(
            [np.ctypeslib.as_array(result.pose_keypoint_scores[i].scores) for i in range(result.n_detections)]
        ),
    )


def pp_segment(output_tensors: List[np.ndarray]) -> Segments:
    """
    Performs post-processing on a Segmentation model result tensor.

    Args:
        output_tensors (List[np.ndarray]): Resulting output tensors to be processed.

    Returns:
        Segments: The post-processed segmentation results.
    """
    return Segments(mask=output_tensors[0])
