import io
import pathlib
import pickle
import typing as t

import pytest
import torch

import smartsim.error as sse
from smartsim._core.mli import workermanager as mli
from smartsim._core.utils import installed_redisai_backends

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


# def test_backend() -> None:
#     """Verify that the worker advertises a backend that it works with"""
#     worker = mli.DefaultTorchWorker

#     exp_backend = "PyTorch"
#     assert worker.backend() == exp_backend


def test_deserialize() -> None:
    """Verify that serialized requests are properly deserialized to
    and converted to the internal representation used by ML workers"""
    worker = mli.SampleTorchWorker
    buffer = io.BytesIO()

    # exp_backend = "TestBackend"
    exp_model_key = "model-key"
    msg = mli.InferenceRequest(model_key=exp_model_key)
    pickle.dump(msg, buffer)

    deserialized: mli.InferenceRequest = worker.deserialize(buffer.getvalue())

    assert deserialized.model_key == exp_model_key
    # assert deserialized.backend == exp_backend


def test_load_model_from_disk(persist_model_file: pathlib.Path) -> None:
    """Verify that a model can be loaded using a FileSystemFeatureStore"""
    worker = mli.SampleTorchWorker
    request = mli.InferenceRequest(raw_model=persist_model_file.read_bytes())

    load_result = worker.load_model(request)

    input = torch.randn(2)
    pred = load_result.model(input)

    assert pred


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_transform_input() -> None:
    """Verify that the default input transform operation is a no-op copy"""
    rows, cols = 1, 4
    num_values = 7
    tensors = [torch.randn((rows, cols)) for _ in range(num_values)]

    request = mli.InferenceRequest()

    inputs: t.List[bytes] = []
    for tensor in tensors:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        inputs.append(buffer.getvalue())

    fetch_result = mli.InputFetchResult(inputs)
    worker = mli.SampleTorchWorker
    result = worker.transform_input(request, fetch_result)
    transformed: t.Collection[torch.Tensor] = result.transformed

    assert len(transformed) == num_values

    for output, expected in zip(transformed, tensors):
        assert output.shape == expected.shape
        assert output.equal(expected)

    transformed = list(transformed)

    original: torch.Tensor = tensors[0]
    assert transformed[0].equal(original)

    # verify a copy was made
    transformed[0] = 2 * transformed[0]
    assert transformed[0].equal(2 * original)


def test_execute_model(persist_model_file: pathlib.Path) -> None:
    """Verify that a model executes corrrectly via the worker"""

    # put model bytes into memory
    model_name = "test-key"
    feature_store = mli.MemoryFeatureStore()
    feature_store[model_name] = persist_model_file.read_bytes()

    worker = mli.SampleTorchWorker
    request = mli.InferenceRequest(model_key=model_name)
    request.raw_model = persist_model_file.read_bytes()
    load_result = worker.load_model(request)

    value = torch.randn(2)
    transform_result = mli.InputTransformResult([value])

    execute_result = worker.execute(request, load_result, transform_result)

    assert execute_result.predictions is not None


def test_execute_missing_model(persist_model_file: pathlib.Path) -> None:
    """Verify that a executing a model with an invalid key fails cleanly"""

    # todo: consider moving to file specific to key tests

    # use key that references an un-set model value
    model_name = "test-key"
    feature_store = mli.MemoryFeatureStore()
    feature_store[model_name] = persist_model_file.read_bytes()

    worker = mli.SampleTorchWorker
    request = mli.InferenceRequest(input_keys=[model_name])

    load_result = mli.ModelLoadResult(None)
    transform_result = mli.InputTransformResult(
        [torch.randn(2), torch.randn(2), torch.randn(2)]
    )

    with pytest.raises(sse.SmartSimError) as ex:
        worker.execute(request, load_result, transform_result)

    assert "Model must be loaded" in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_transform_output() -> None:
    """Verify that the default output transform operation is a no-op copy"""
    rows, cols = 1, 4
    num_values = 7
    inputs = [torch.randn((rows, cols)) for _ in range(num_values)]
    exp_outputs = [torch.Tensor(tensor) for tensor in inputs]

    worker = mli.SampleTorchWorker
    request = mli.InferenceRequest()
    exec_result = mli.ExecuteResult(inputs)

    result = worker.transform_output(
        request, exec_result
    )
    # transform_result = mli.InputTransformResult(transformed)

    assert len(result.outputs) == num_values

    for output, expected in zip(result.outputs, exp_outputs):
        assert output.shape == expected.shape
        assert output.equal(expected)

    transformed = list(result.outputs)

    # verify a copy was made
    original: torch.Tensor = inputs[0]
    transformed[0] = 2 * transformed[0]

    assert transformed[0].equal(2 * original)
