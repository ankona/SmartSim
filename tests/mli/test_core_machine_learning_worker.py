import io
import pathlib
import pytest
import time
import torch
import typing as t

from smartsim import workermanager as mli
from smartsim._core.utils import installed_redisai_backends

import smartsim.error as sse

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
    key = mli.FileSystemKey(persist_model_file)

    raw_bytes = worker.fetch_model(key)
    assert raw_bytes
    assert raw_bytes == persist_model_file.read_bytes()


def test_fetch_model_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    bad_key_path = pathlib.Path("/path/that/doesnt/exist")
    key = mli.FileSystemKey(bad_key_path)

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(key)

    # ensure the error message includes key-identifying information
    assert str(bad_key_path) in ex.value.args[0]


def test_fetch_model_feature_store(persist_model_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    model_name = "test-model"
    feature_store = mli.DictFeatureStore()

    # todo: consider if this abstraction as reversed. should the FS instead give
    # out keys instead of giving an FS to the key?

    # create a key to retrieve from the feature store
    key = mli.FeatureStoreKey(model_name, feature_store)
    # put model bytes into the feature store
    model_bytes = persist_model_file.read_bytes()
    key.put(model_bytes)

    raw_bytes = worker.fetch_model(key)
    assert raw_bytes
    assert raw_bytes == persist_model_file.read_bytes()


def test_fetch_model_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (feature store) key"""
    worker = mli.MachineLearningWorkerCore

    bad_key = "some-key"
    feature_store = mli.DictFeatureStore()
    key = mli.FeatureStoreKey(bad_key, feature_store)

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(key)

    # ensure the error message includes key-identifying information
    assert bad_key in ex.value.args[0]


def test_fetch_model_memory(persist_model_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    model_name = "test-model"
    key = mli.MemoryKey(model_name, persist_model_file.read_bytes())

    raw_bytes = worker.fetch_model(key)
    assert raw_bytes
    assert raw_bytes == persist_model_file.read_bytes()


def test_fetch_input_disk(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore
    key = mli.FileSystemKey(persist_tensor_file)

    raw_bytes = worker.fetch_inputs([key])
    assert raw_bytes
    # assert raw_bytes == persist_tensor_file.read_bytes()


def test_fetch_input_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    bad_key_path = pathlib.Path("/path/that/doesnt/exist")
    key = mli.FileSystemKey(bad_key_path)

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs([key])

    # ensure the error message includes key-identifying information
    assert str(bad_key_path) in ex.value.args[0]


def test_fetch_input_feature_store(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (feature store) key"""
    worker = mli.MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = mli.DictFeatureStore()

    # todo: consider if this abstraction as reversed. should the FS instead give
    # out keys instead of giving an FS to the key?

    # create a key to retrieve from the feature store
    key = mli.FeatureStoreKey(tensor_name, feature_store)
    # put model bytes into the feature store
    input_bytes = persist_tensor_file.read_bytes()
    key.put(input_bytes)

    raw_bytes = worker.fetch_inputs([key])
    assert raw_bytes
    assert raw_bytes[0][:10] == persist_tensor_file.read_bytes()[:10]


def test_fetch_multi_input_feature_store(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves multiple tensor/input
    when given a valid collection of (feature store) keys"""
    worker = mli.MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = mli.DictFeatureStore()

    # todo: consider if this abstraction as reversed. should the FS instead give
    # out keys instead of giving an FS to the key?

    # create a key to retrieve from the feature store
    key = mli.FeatureStoreKey(tensor_name, feature_store)
    # put model bytes into the feature store
    input_bytes = persist_tensor_file.read_bytes()
    key.put(input_bytes)

    body2 = b"abcdefghijklmnopqrstuvwxyz"
    key2 = mli.FeatureStoreKey(tensor_name + "2", feature_store)
    key2.put(body2)

    body3 = b"mnopqrstuvwxyzabcdefghijkl"
    key3 = mli.FeatureStoreKey(tensor_name + "3", feature_store)
    key3.put(body3)

    raw_bytes = worker.fetch_inputs([key, key2, key3])
    assert raw_bytes
    assert raw_bytes[0][:10] == persist_tensor_file.read_bytes()[:10]
    assert raw_bytes[1][:10] == body2[:10]
    assert raw_bytes[2][:10] == body3[:10]


def test_fetch_input_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (feature store) key"""
    worker = mli.MachineLearningWorkerCore

    bad_key = "some-key"
    feature_store = mli.DictFeatureStore()
    key = mli.FeatureStoreKey(bad_key, feature_store)

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs([key])

    # ensure the error message includes key-identifying information
    assert bad_key in ex.value.args[0]


def test_fetch_input_memory(persist_tensor_file: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    worker = mli.MachineLearningWorkerCore

    model_name = "test-model"
    key = mli.MemoryKey(model_name, persist_tensor_file.read_bytes())

    raw_bytes = worker.fetch_inputs([key])
    assert raw_bytes
    # assert raw_bytes == persist_tensor_file.read_bytes()


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
    feature_store = mli.DictFeatureStore()

    # create a key to retrieve from the feature store
    key1 = mli.FeatureStoreKey(key_name + "1", feature_store)
    key2 = mli.FeatureStoreKey(key_name + "2", feature_store)
    key3 = mli.FeatureStoreKey(key_name + "3", feature_store)
    exp_keys = key1, key2, key3

    keys = key_name + "1", key_name + "2", key_name + "3"
    data = b"abcdef", b"ghijkl", b"mnopqr"

    output_keys = worker.place_output(keys, data, feature_store)
    assert len(output_keys) == len(keys)

    for i, exp_key in enumerate(exp_keys):
        assert exp_key.retrieve() == data[i]

    for i in range(3):
        assert feature_store[keys[i]] == data[i]


# @pytest.fixture
# def persist_model_file(test_dir: str) -> pathlib.Path:
#     test_path = pathlib.Path(test_dir)
#     model_path = test_path / "basic.pt"

#     model = torch.nn.Linear(2, 1)
#     torch.save(model, model_path)

#     return model_path


# def test_backend() -> None:
#     """Verify that the worker advertises a backend that it works with"""
#     worker = mli.DefaultTorchWorker

#     exp_backend = "PyTorch"
#     assert worker.backend() == exp_backend


# def test_deserialize() -> None:
#     """Verify that tensors are properly deserialized"""
#     worker = mli.DefaultTorchWorker

#     tensor = torch.randn(42)
#     buffer = io.BytesIO()
#     torch.save(tensor, buffer)

#     deserialized: torch.Tensor = worker.deserialize(buffer.getvalue())

#     assert tensor.equal(deserialized)


# def test_load_model_from_disk(persist_model_file: pathlib.Path) -> None:
#     """Verify that a model can be loaded using a FileSystemKey"""
#     worker = mli.DefaultTorchWorker
#     key = mli.FileSystemKey(persist_model_file)
#     model_ref = mli.MachineLearningModelRef(worker.backend(), key)

#     model = worker.load_model(model_ref)

#     input = torch.randn(2)
#     pred = model(input)

#     assert pred


# def test_load_model_from_feature_store(persist_model_file: pathlib.Path) -> None:
#     """Verify that a model can be loaded using a FileSystemKey"""

#     model_name = "test-model"
#     feature_store = mli.DictFeatureStore()

#     # create a key to retrieve from the feature store
#     key = mli.FeatureStoreKey(model_name, feature_store)
#     # put model bytes into the feature store
#     key.put(persist_model_file.read_bytes())

#     worker = mli.DefaultTorchWorker
#     model_ref = mli.MachineLearningModelRef(worker.backend(), key)

#     model = worker.load_model(model_ref)

#     input = torch.randn(2)
#     pred = model(input)

#     assert pred


# def test_load_model_from_memory(persist_model_file: pathlib.Path) -> None:
#     """Verify that a model can be loaded using a MemoryKey"""

#     # put model bytes into memory
#     key = mli.MemoryKey("test-key", persist_model_file.read_bytes())

#     worker = mli.DefaultTorchWorker
#     model_ref = mli.MachineLearningModelRef(worker.backend(), key)

#     model = worker.load_model(model_ref)

#     input = torch.randn(2)
#     pred = model(input)

#     assert pred


# # @pytest.mark.skipif(not is_dragon, reason="Test is only for Dragon WLM systems")
# def test_load_model_from_dragon(persist_model_file: pathlib.Path) -> None:
#     """Verify that a model can be loaded using a key ref to dragon feature store"""

#     model_name = "test-model"
#     storage = (
#         mli.DragonDict()
#     )  # todo: use _real_ DragonDict instead of mock & re-enable skipif
#     feature_store = mli.DragonFeatureStore(storage)

#     # create a key to retrieve from the feature store
#     key = mli.FeatureStoreKey(model_name, feature_store)
#     # put model bytes into the feature store
#     key.put(persist_model_file.read_bytes())

#     worker = mli.DefaultTorchWorker
#     model_ref = mli.MachineLearningModelRef(worker.backend(), key)

#     model = worker.load_model(model_ref)

#     input = torch.randn(2)
#     pred = model(input)

#     assert pred


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_transform_input() -> None:
#     """Verify that the default input transform operation is a no-op copy"""
#     rows, cols = 1, 4
#     num_values = 7
#     inputs = [torch.randn((rows, cols)) for _ in range(num_values)]
#     exp_outputs = [torch.Tensor(tensor) for tensor in inputs]

#     worker = mli.DefaultTorchWorker
#     transformed: t.Collection[torch.Tensor] = worker.transform_input(inputs)

#     assert len(transformed) == num_values

#     for output, expected in zip(transformed, exp_outputs):
#         assert output.shape == expected.shape
#         assert output.equal(expected)

#     # verify a copy was made
#     original: torch.Tensor = inputs[0]
#     transformed[0] = 2 * transformed[0]

#     assert transformed[0].equal(2 * original)


# def test_execute_model(persist_model_file: pathlib.Path) -> None:
#     """Verify that a model executes corrrectly via the worker"""

#     # put model bytes into memory
#     key = mli.MemoryKey("test-key", persist_model_file.read_bytes())

#     worker = mli.DefaultTorchWorker
#     model_ref = mli.MachineLearningModelRef(worker.backend(), key)

#     # model = worker.load_model(model_ref)
#     input = torch.randn(2)

#     pred: t.List[torch.Tensor] = worker.execute(model_ref, [input])

#     assert pred


# def test_execute_missing_model(persist_model_file: pathlib.Path) -> None:
#     """Verify that a executing a model with an invalid key fails cleanly"""

#     # todo: consider moving to file specific to key tests

#     # use key that references an un-set model value
#     key = mli.MemoryKey("test-key", b"")

#     worker = mli.DefaultTorchWorker
#     model_ref = mli.MachineLearningModelRef(worker.backend(), key)

#     with pytest.raises(ValueError) as ex:
#         model_ref.model()

#     assert "empty value" in ex.value.args[0]


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_transform_output() -> None:
#     """Verify that the default output transform operation is a no-op copy"""
#     rows, cols = 1, 4
#     num_values = 7
#     inputs = [torch.randn((rows, cols)) for _ in range(num_values)]
#     exp_outputs = [torch.Tensor(tensor) for tensor in inputs]

#     worker = mli.DefaultTorchWorker
#     transformed: t.Collection[torch.Tensor] = worker.transform_output(inputs)

#     assert len(transformed) == num_values

#     for output, expected in zip(transformed, exp_outputs):
#         assert output.shape == expected.shape
#         assert output.equal(expected)

#     # verify a copy was made
#     original: torch.Tensor = inputs[0]
#     transformed[0] = 2 * transformed[0]

#     assert transformed[0].equal(2 * original)
