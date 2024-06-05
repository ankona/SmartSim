import io
import pathlib
import time
import typing as t

import pytest
import torch

import smartsim.error as sse
from smartsim import workermanager as mli
from smartsim._core.utils import installed_redisai_backends

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_b

# retrieved from pytest fixtures
is_dragon = pytest.test_launcher == "dragon"
torch_available = "torch" in installed_redisai_backends()


@pytest.fixture
def persist_model_file(test_dir: str) -> pathlib.Path:
    ts_start = time.time_ns()
    print("Starting model file creation...")
    test_path = pathlib.Path(test_dir)
    model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)
    ts_end = time.time_ns()

    ts_elapsed = (ts_end - ts_start) / 1000000000
    print(f"Model file creation took {ts_elapsed} seconds")
    return model_path


@pytest.fixture
def persist_tensor_file(test_dir: str) -> pathlib.Path:
    ts_start = time.time_ns()
    print("Starting model file creation...")
    test_path = pathlib.Path(test_dir)
    file_path = test_path / "tensor.pt"

    tensor = torch.randn((100, 100, 2))
    torch.save(tensor, file_path)
    ts_end = time.time_ns()

    ts_elapsed = (ts_end - ts_start) / 1000000000
    print(f"Tensor file creation took {ts_elapsed} seconds")
    return file_path


def test_fetch_model_disk(persist_model_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore
    key = str(persist_model_file)
    feature_store = mli.FileSystemFeatureStore()
    feature_store[str(persist_model_file)] = persist_model_file.read_bytes()

    raw_bytes = worker.fetch_model(key, feature_store)
    assert raw_bytes
    assert raw_bytes == persist_model_file.read_bytes()


def test_fetch_model_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (file system) key"""
    worker = mli.MachineLearningWorkerCore
    feature_store = mli.FileSystemFeatureStore()

    bad_key_path = pathlib.Path("/path/that/doesnt/exist")

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(str(bad_key_path), feature_store)

    # ensure the error message includes key-identifying information
    assert str(bad_key_path) in ex.value.args[0]


def test_fetch_model_feature_store(persist_model_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    # create a key to retrieve from the feature store
    key = "test-model"

    # put model bytes into the feature store
    feature_store = mli.MemoryFeatureStore()
    feature_store[key] = persist_model_file.read_bytes()

    raw_bytes = worker.fetch_model(key, feature_store)
    assert raw_bytes
    assert raw_bytes == persist_model_file.read_bytes()


def test_fetch_model_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (feature store) key"""
    worker = mli.MachineLearningWorkerCore

    bad_key = "some-key"
    feature_store = mli.MemoryFeatureStore()

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(bad_key, feature_store)

    # ensure the error message includes key-identifying information
    assert bad_key in ex.value.args[0]


def test_fetch_model_memory(persist_model_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    key = "test-model"
    feature_store = mli.MemoryFeatureStore()
    feature_store[key] = persist_model_file.read_bytes()

    raw_bytes = worker.fetch_model(key, feature_store)
    assert raw_bytes
    assert raw_bytes == persist_model_file.read_bytes()


def test_fetch_input_disk(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    tensor_name = str(persist_tensor_file)

    request = mli.InferenceRequest(input_keys=[tensor_name])
    worker = mli.MachineLearningWorkerCore

    feature_store = mli.MemoryFeatureStore()
    feature_store[tensor_name] = persist_tensor_file.read_bytes()

    raw_bytes = worker.fetch_inputs(request, feature_store)
    assert raw_bytes


def test_fetch_input_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    key = "/path/that/doesnt/exist"
    feature_store = mli.MemoryFeatureStore()

    request = mli.InferenceRequest(input_keys=[key])

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs(request, feature_store)

    # ensure the error message includes key-identifying information
    assert key in ex.value.args[0]


def test_fetch_input_feature_store(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (feature store) key"""
    worker = mli.MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = mli.MemoryFeatureStore()

    request = mli.InferenceRequest(input_keys=[tensor_name])

    # todo: consider if this abstraction as reversed. should the FS instead give
    # out keys instead of giving an FS to the key?

    # put model bytes into the feature store
    feature_store[tensor_name] = persist_tensor_file.read_bytes()

    raw_bytes = worker.fetch_inputs(request, feature_store)
    assert raw_bytes
    assert list(raw_bytes)[0][:10] == persist_tensor_file.read_bytes()[:10]


def test_fetch_multi_input_feature_store(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves multiple tensor/input
    when given a valid collection of (feature store) keys"""
    worker = mli.MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = mli.MemoryFeatureStore()

    # put model bytes into the feature store
    body1 = persist_tensor_file.read_bytes()
    feature_store[tensor_name + "1"] = body1

    body2 = b"abcdefghijklmnopqrstuvwxyz"
    feature_store[tensor_name + "2"] = body2

    body3 = b"mnopqrstuvwxyzabcdefghijkl"
    feature_store[tensor_name + "3"] = body3

    request = mli.InferenceRequest(
        input_keys=[tensor_name + "1", tensor_name + "2", tensor_name + "3"]
    )

    raw_bytes = worker.fetch_inputs(request, feature_store)

    raw_bytes = list(raw_bytes)
    assert raw_bytes
    assert raw_bytes[0][:10] == persist_tensor_file.read_bytes()[:10]
    assert raw_bytes[1][:10] == body2[:10]
    assert raw_bytes[2][:10] == body3[:10]


def test_fetch_input_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (feature store) key"""
    worker = mli.MachineLearningWorkerCore

    bad_key = "some-key"
    feature_store = mli.MemoryFeatureStore()
    request = mli.InferenceRequest(input_keys=[bad_key])

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs(request, feature_store)

    # ensure the error message includes key-identifying information
    assert bad_key in ex.value.args[0]


def test_fetch_input_memory(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore
    feature_store = mli.MemoryFeatureStore()

    model_name = "test-model"
    feature_store[model_name] = persist_tensor_file.read_bytes()
    request = mli.InferenceRequest(input_keys=[model_name])

    raw_bytes = worker.fetch_inputs(request, feature_store)
    assert raw_bytes


def test_batch_requests() -> None:
    """Verify batch requests handles an empty data set gracefully"""
    worker = mli.MachineLearningWorkerCore

    with pytest.raises(NotImplementedError):
        # NOTE: we expect this to fail since it's not yet implemented.
        # TODO: once implemented, replace this expectation of failure...
        worker.batch_requests([], 10)


def test_place_outputs() -> None:
    """Verify outputs are shared using the feature store"""
    worker = mli.MachineLearningWorkerCore

    key_name = "test-model"
    feature_store = mli.MemoryFeatureStore()

    # create a key to retrieve from the feature store
    keys = key_name + "1", key_name + "2", key_name + "3"
    data = b"abcdef", b"ghijkl", b"mnopqr"

    for k, v in zip(keys, data):
        feature_store[k] = v

    output_keys = worker.place_output(keys, data, feature_store)
    assert len(output_keys) == len(keys)

    for i in range(3):
        assert feature_store[keys[i]] == data[i]
