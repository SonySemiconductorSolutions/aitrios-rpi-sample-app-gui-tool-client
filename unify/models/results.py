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

import base64
import gzip
from typing import Iterator, List, Tuple, Union

import numpy as np


class Classifications:
    """
    Class representing classification results.

    Attributes:
    - confidence (np.ndarray): Array of shape (n,) representing the confidence of N detections.
    - class_id (np.ndarray): Array of shape (n,) representing the class id of N detections.
    """

    confidence: np.ndarray
    class_id: np.ndarray

    def __init__(
        self,
        confidence=np.empty((0,)),
        class_id=np.empty((0,)),
    ) -> None:
        """
        Initialize a new instance of Classifications.

        Args:
        - confidence (np.ndarray): Array of shape (n,) representing the confidence of N detections.
        - class_id (np.ndarray): Array of shape (n,) representing the class id of N detections.
        """
        self.confidence = confidence
        self.class_id = class_id

    def __len__(self):
        """
        Returns the number of detections.
        """
        return len(self.class_id)

    def __copy__(self):
        """
        Returns a copy of the current detections.
        """
        new_instance = Classifications()
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)

        return new_instance

    def copy(self):
        """
        Returns a copy of the current detections.
        """
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Classifications":
        """
        Returns a new Classifications object with the selected detections.
        Could be a subsection of the current detections.

        Args:
        - index (int, slice, List[int], np.ndarray): The index or indices of the detections to select.

        Returns:
        - Classifications: A new Classifications object with the selected detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        return res

    def __iter__(self) -> Iterator[Tuple[float, int]]:
        """
        Iterate over the detections.

        Yields:
        - Tuple[float, int]: A tuple containing the confidence and class id of each detection.
        """
        for i in range(len(self)):
            yield (
                self.confidence[i],
                self.class_id[i],
            )

    def __add__(self, other: "Classifications") -> "Classifications":
        """
        Concatenate two Classifications objects.

        Args:
        - other (Classifications): The other Classifications object to concatenate.

        Returns:
        - Classifications: The concatenated Classifications.
        """
        if not isinstance(other, Classifications):
            raise TypeError(f"Unsupported operand type(s) for +: 'Classifications' and '{type(other)}'")

        result = self.copy()
        result.confidence = np.concatenate([self.confidence, other.confidence])
        result.class_id = np.concatenate([self.class_id, other.class_id])
        return result

    def __str__(self) -> str:
        """
        Return a string representation of the Classifications object.

        Returns:
        - str: A string representation of the Classifications object.
        """
        return f"Classifications(class_id:\t {self.class_id}, \tconfidence:\t {self.confidence})"

    def json(self) -> dict:
        """
        Convert the Classifications object to a JSON-serializable dictionary.

        Returns:
        - dict: A dictionary representation of the Classifications object with the following keys:
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class IDs.
        """
        return {
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
        }


class Detections:
    """
    Class representing object detections.

    Attributes:
    - bbox (np.ndarray): Array of shape (n, 4) the bounding boxes [x1, y1, x2, y2] of N detections
    - confidence (np.ndarray): Array of shape (n,) the confidence of N detections
    - class_id (np.ndarray): Array of shape (n,) the class id of N detections
    - tracker_id (np.ndarray): Array of shape (n,) the tracker id of N detections

    Properties:
    - area (np.ndarray): Array of shape (n,) the area of the bounding boxes of N detections
    - bbox_width (np.ndarray): Array of shape (n,) the width of the bounding boxes of N detections
    - bbox_height (np.ndarray): Array of shape (n,) the height of the bounding boxes of N detections
    """

    bbox: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: np.ndarray

    def __init__(
        self,
        bbox=np.empty((0, 4)),
        confidence=np.empty((0,)),
        class_id=np.empty((0,)),
    ) -> None:
        """
        Initialize the Detections object.

        Args:
        - bbox (np.ndarray): Array of shape (n, 4) the bounding boxes [x1, y1, x2, y2] of N detections
        - confidence (np.ndarray): Array of shape (n,) the confidence of N detections
        - class_id (np.ndarray): Array of shape (n,) the class id of N detections
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = np.empty((len(class_id),))

    def __len__(self):
        """
        Returns the number of detections.
        """
        return len(self.class_id)

    def __copy__(self):
        """
        Returns a copy of the current detections.
        """
        new_instance = Detections()
        new_instance.bbox = np.copy(self.bbox)
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)
        new_instance.tracker_id = np.copy(self.tracker_id)

        return new_instance

    def copy(self):
        """
        Returns a copy of the current detections.
        """
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Detections":
        """
        Returns a new Detections object with the selected detections.

        Args:
            index (Union[int, slice, List[int], np.ndarray]): The index or indices of the detections to select.

        Returns:
            Detections: A new Detections object with the selected detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        res.bbox = self.bbox[index] if self.bbox is not None else None
        res.tracker_id = self.tracker_id[index] if self.tracker_id is not None else None
        return res

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int]]:
        for i in range(len(self)):
            yield (
                self.bbox[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i],
            )

    def __add__(self, other: "Detections") -> "Detections":
        """
        Concatenate two Detections objects.

        Args:
        - other (Detections): The other Detections object to concatenate.

        Returns:
        - Detections: The concatenated Detections.
        """
        if not isinstance(other, Detections):
            raise TypeError(f"Unsupported operand type(s) for +: 'Detections' and '{type(other)}'")

        result = self.copy()
        result.bbox = np.vstack((result.bbox, other.bbox))
        result.confidence = np.concatenate([self.confidence, other.confidence])
        result.class_id = np.concatenate([self.class_id, other.class_id])
        result.tracker_id = np.concatenate([self.tracker_id, other.tracker_id])
        return result

    def __str__(self) -> str:
        """
        Return a string representation of the Detections object.

        Returns:
        - str: A string representation of the Detections object.
        """
        s = f"Detections(class_id:\t {self.class_id}, \tconfidence:\t {self.confidence}, \tbbox_shape: {self.bbox.shape}"
        if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape:
            s += f", \ttrack_ids:\t {self.tracker_id}"
        return s + ")"

    def json(self) -> dict:
        """
        Convert the Detections object to a JSON-serializable dictionary.

        Returns:
        - dict: A dictionary representation of the Detections object with the following keys:
            - "bbox" (list): The bounding box coordinates.
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class IDs.
            - "tracker_id" (list or None): The tracker IDs, or None if tracker_id is not set or its shape does not match.
        """
        return {
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
            "tracker_id": (
                self.tracker_id.tolist()
                if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape
                else None
            ),
        }

    # PROPERTIES
    @property
    def area(self) -> np.ndarray:
        """
        Get the area of each bounding box.

        Returns:
            np.ndarray: An array containing the area of each bounding box.
        """
        widths = self.bbox[:, 2] - self.bbox[:, 0]
        heights = self.bbox[:, 3] - self.bbox[:, 1]
        return widths * heights

    @property
    def bbox_width(self) -> np.ndarray:
        """
        Get the width of the bounding boxes.

        Returns:
            np.ndarray: An array containing the width of each bounding box.
        """
        return self.bbox[:, 2] - self.bbox[:, 0]

    @property
    def bbox_height(self) -> np.ndarray:
        """
        Get the height of the bounding boxes.

        Returns:
            np.ndarray: An array containing the height of each bounding box.
        """
        return self.bbox[:, 3] - self.bbox[:, 1]


class Poses:
    """
    Data class for pose estimation results.

    Attributes:
    - n_detections (int): Number of detected poses.
    - scores (np.ndarray):
    - keypoints (np.ndarray):
    - keypoint_scores (np.ndarray):
    """

    n_detections: int
    scores: np.ndarray
    keypoints: np.ndarray
    keypoint_scores: np.ndarray

    def __init__(
        self,
        n_detections=0,
        scores=np.empty((0,)),
        keypoints=np.empty((0,)),
        keypoint_scores=np.empty((0,)),
    ) -> None:

        self.n_detections = n_detections
        self.scores = scores
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores

    def __str__(self) -> str:
        """
        Return a string representation of the Poses object.

        Returns:
        - str: A string representation of the Poses object.
        """
        return (
            f"Poses(n_detections: {self.n_detections}, "
            f"scores: {self.scores}, "
            f"keypoints: {self.keypoints}, "
            f"keypoint_scores: {self.keypoint_scores})"
        )

    def json(self) -> dict:
        """
        Convert the Detections object to a JSON-serializable dictionary.

        Returns:
        - dict: A dictionary representation of the Detections object with the following keys:
            - "n_detections" (int): Number of detected poses.
            - "scores" (list):
            - "keypoints" (list):
            - "keypoint_scores" (list):
        """
        return {
            "n_detections": self.n_detections,
            "scores": self.scores.tolist(),
            "keypoints": self.keypoints.tolist(),
            "keypoint_scores": self.keypoint_scores.tolist(),
        }


class Segments:
    """
    Data class for segmentation results.

    Attributes:
    - mask (np.ndarray): Mask arrays containing the id for each identified segment.

    Properties:
    - n_segments (int): Number of detected segments.
    - indeces (List[int]): The list of indeces of the detected segments.
    """

    mask: np.ndarray

    def __init__(self, mask=np.empty((0,))) -> None:
        self.mask = mask.astype(np.uint8)

    @property
    def n_segments(self) -> int:
        """
        Return the number found segments, while ignore the background.
        """
        return len(self.indeces)

    @property
    def indeces(self) -> List[int]:
        """
        Return the found indeces in the mask and ignore the background (id: 0).
        """
        found_indices = np.unique(self.mask)
        return found_indices[found_indices != 0]

    def get_mask(self, id: int):
        """
        Returns the mask of a specific index.
        """
        return (self.mask == id).astype(np.uint8)

    def __str__(self) -> str:
        """
        Return a string representation of the Segments object.

        Returns:
        - str: A string representation of the Segments object.
        """
        return f"Segments(n_segments: {self.n_segments}, " f"indeces: {self.indeces}" f"mask: {self.mask})"

    def json(self) -> dict:
        """
        Convert the Segments object to a JSON-serializable dictionary.

        Returns:
        - dict: A dictionary representation of the Segments object with the following keys:
            - "n_segments" (int): Number of detected segments.
            - "indeces" (list): List of the index corresponding to each segment.
            - "masks" (list): List of mask arrays for each segment (compressed and base64 encoded).
        """
        return {
            "n_segments": self.n_segments,
            "indeces": self.indeces.tolist(),
            "mask": base64.b64encode(gzip.compress(self.mask.tobytes())).decode("utf-8"),
        }
