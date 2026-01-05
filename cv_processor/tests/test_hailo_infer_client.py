from unittest.mock import MagicMock

from cv_processor.hailo_infer_client import HailoInferClient


def test_client_uses_injected_infer():
    infer_cls = MagicMock()
    infer_instance = infer_cls.return_value
    infer_instance.get_input_shape.return_value = (640, 640, 3)

    client = HailoInferClient("net.hef", 2, infer_cls=infer_cls)
    assert client.get_input_shape() == (640, 640, 3)

    callback = MagicMock()
    client.run(["frame"], callback)
    client.close()

    infer_cls.assert_called_once_with("net.hef", 2)
    infer_instance.run.assert_called_once_with(["frame"], callback)
    infer_instance.close.assert_called_once()
