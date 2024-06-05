import io
import pathlib
import pickle
import typing as t

import pytest
import time
import torch

import smartsim.error as sse
from smartsim._core.mli import workermanager as mli
from smartsim._core.utils import installed_redisai_backends
from smartsim._core.mli.message_handler import MessageHandler

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_b

# retrieved from pytest fixtures
is_dragon = pytest.test_launcher == "dragon"
torch_available = "torch" in installed_redisai_backends()


@pytest.fixture
def persist_model_file(test_dir: str) -> pathlib.Path:
    test_path = pathlib.Path(test_dir)
    model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


def test_deserialize(test_dir: str, persist_model_file: pathlib.Path) -> None:
    """Verify that the MessageHandler deserializer handles the
    message properly"""
    worker = mli.IntegratedTorchWorker
    # buffer = io.BytesIO()

    # timestamp = time.time_ns()
    # test_path = pathlib.Path(test_dir)
    input_tensor = torch.randn(2)
    # mock_channel = test_path / f"brainstorm-{timestamp}.txt"

    expected_device = "cpu"
    expected_callback_channel = b"faux_channel_descriptor_bytes"

    message_tensor = MessageHandler.build_tensor(input_tensor, "c", "float32", [2])
    message_tensor_key = MessageHandler.build_tensor_key("demo")
    request = MessageHandler.build_request(
        expected_callback_channel,
        persist_model_file.read_bytes(),
        expected_device,
        [message_tensor],
        [message_tensor_key],
        None,
    )

    # proto_guy_dictionary = request.to_dict()

    msg_bytes = MessageHandler.serialize_request(request)

    inference_request = worker.deserialize(msg_bytes)
    assert inference_request.device == expected_device
    assert inference_request.callback._descriptor == expected_callback_channel


def test_serialize(test_dir: str, persist_model_file: pathlib.Path) -> None:
    """Verify that the worker correctly executes reply serialization"""
    worker = mli.IntegratedTorchWorker

    reply = mli.InferenceReply()
    reply.output_keys = ["foo", "bar"]

    # use the worker implementation of reply serialization to get bytes for
    # use on the callback channel
    reply_bytes = worker.serialize_reply(reply)
    assert reply_bytes is not None

    # deserialize to verity the mapping in the worker.serialize_reply was correct
    actual_reply = MessageHandler.deserialize_response(reply_bytes)

    actual_tensor_keys = [tk.key for tk in actual_reply.result.keys]
    assert set(actual_tensor_keys) == set(reply.output_keys)
    assert actual_reply.status == 200
    assert actual_reply.statusMessage == "Inference Complete"
