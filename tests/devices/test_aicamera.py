import pytest

from unify.devices import AiCamera


@pytest.fixture
@pytest.mark.aicam
def test_device():
    device = AiCamera(headless=True, timeout=5)
    # model = ...
    # device.deploy(model)
    return device


@pytest.mark.aicam
def test_capture(test_device):

    with test_device as stream:
        for frame in stream:
            print("Frame: ", frame.fps)
