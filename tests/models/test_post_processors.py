import numpy as np

from unify.models import Classifications, Detections
from unify.models.post_processors import pp_cls, pp_od_bcsn, pp_od_bscn


def test_pp_cls():
    output_tensor = [np.array([0.1, 0.4, 0.3, 0.2])]
    result = pp_cls(output_tensor)

    assert isinstance(result, Classifications)
    assert np.array_equal(result.confidence, np.array([0.4, 0.3, 0.2, 0.1]))
    assert np.array_equal(result.class_id, np.array([1, 2, 3, 0]))


def test_pp_od_bcsn():
    
    output_tensors = [
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]), # Box coordinates
        np.array([1, 2]),       # Class IDs for 2 detections
        np.array([0.9, 0.8]),   # Scores for 2 detections
        np.array([2])           # Number of detections
    ]

    result = pp_od_bcsn(output_tensors)

    assert isinstance(result, Detections)
    assert np.array_equal(result.bbox, np.array([[0.1, 0.0, 0.3, 0.2], [0.5, 0.4, 0.7, 0.6]]))
    assert np.array_equal(result.class_id, np.array([1, 2]))
    assert np.array_equal(result.confidence, np.array([0.9, 0.8]))


def test_pp_od_bscn():
    
    output_tensors = [
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]), # Box coordinates
        np.array([0.9, 0.8]),   # Scores for 2 detections
        np.array([1, 2]),       # Class IDs for 2 detections
        np.array([2])           # Number of detections
    ]
   
    result = pp_od_bscn(output_tensors)

    assert isinstance(result, Detections)
    assert np.array_equal(result.bbox, np.array([[0.1, 0.0, 0.3, 0.2], [0.5, 0.4, 0.7, 0.6]]))
    assert np.array_equal(result.confidence, np.array([0.9, 0.8]))
    assert np.array_equal(result.class_id, np.array([1, 2]))