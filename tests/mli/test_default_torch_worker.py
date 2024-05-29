import io
import pathlib
import pytest
import torch
import typing as t

from smartsim import brainstorm2 as mli
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


def test_backend() -> None:
    """Verify that the worker advertises a backend that it works with"""
    worker = mli.DefaultTorchWorker

    exp_backend = "PyTorch"
    assert worker.backend() == exp_backend


def test_deserialize() -> None:
    """Verify that tensors are properly deserialized"""
    worker = mli.DefaultTorchWorker

    tensor = torch.randn(42)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)

    deserialized: torch.Tensor = worker.deserialize(buffer.getvalue())

    assert tensor.equal(deserialized)


def test_load_model_from_disk(persist_model_file: pathlib.Path) -> None:
    """Verify that a model can be loaded using a FileSystemKey"""
    worker = mli.DefaultTorchWorker
    key = mli.FileSystemKey(persist_model_file)
    model_ref = mli.MachineLearningModelRef(worker.backend(), key)

    model = worker.load_model(model_ref)

    input = torch.randn(2)
    pred = model(input)

    assert pred


def test_load_model_from_feature_store(persist_model_file: pathlib.Path) -> None:
    """Verify that a model can be loaded using a FileSystemKey"""

    model_name = "test-model"
    feature_store = mli.DictFeatureStore()

    # create a key to retrieve from the feature store
    key = mli.FeatureStoreKey(model_name, feature_store)
    # put model bytes into the feature store
    key.put(persist_model_file.read_bytes())

    worker = mli.DefaultTorchWorker
    model_ref = mli.MachineLearningModelRef(worker.backend(), key)

    model = worker.load_model(model_ref)

    input = torch.randn(2)
    pred = model(input)

    assert pred


def test_load_model_from_memory(persist_model_file: pathlib.Path) -> None:
    """Verify that a model can be loaded using a MemoryKey"""

    # put model bytes into memory
    key = mli.MemoryKey("test-key", persist_model_file.read_bytes())

    worker = mli.DefaultTorchWorker
    model_ref = mli.MachineLearningModelRef(worker.backend(), key)

    model = worker.load_model(model_ref)

    input = torch.randn(2)
    pred = model(input)

    assert pred


# @pytest.mark.skipif(not is_dragon, reason="Test is only for Dragon WLM systems")
def test_load_model_from_dragon(persist_model_file: pathlib.Path) -> None:
    """Verify that a model can be loaded using a key ref to dragon feature store"""

    model_name = "test-model"
    storage = (
        mli.DragonDict()
    )  # todo: use _real_ DragonDict instead of mock & re-enable skipif
    feature_store = mli.DragonFeatureStore(storage)

    # create a key to retrieve from the feature store
    key = mli.FeatureStoreKey(model_name, feature_store)
    # put model bytes into the feature store
    key.put(persist_model_file.read_bytes())

    worker = mli.DefaultTorchWorker
    model_ref = mli.MachineLearningModelRef(worker.backend(), key)

    model = worker.load_model(model_ref)

    input = torch.randn(2)
    pred = model(input)

    assert pred


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_transform_input() -> None:
    """Verify that the default input transform operation is a no-op copy"""
    rows, cols = 1, 4
    num_values = 7
    inputs = [torch.randn((rows, cols)) for _ in range(num_values)]
    exp_outputs = [torch.Tensor(tensor) for tensor in inputs]

    worker = mli.DefaultTorchWorker
    transformed: t.Collection[torch.Tensor] = worker.transform_input(inputs)

    assert len(transformed) == num_values

    for output, expected in zip(transformed, exp_outputs):
        assert output.shape == expected.shape
        assert output.equal(expected)

    # verify a copy was made
    original: torch.Tensor = inputs[0]
    transformed[0] = 2 * transformed[0]

    assert transformed[0].equal(2 * original)


def test_execute_model(persist_model_file: pathlib.Path) -> None:
    """Verify that a model executes corrrectly via the worker"""

    # put model bytes into memory
    key = mli.MemoryKey("test-key", persist_model_file.read_bytes())

    worker = mli.DefaultTorchWorker
    model_ref = mli.MachineLearningModelRef(worker.backend(), key)

    # model = worker.load_model(model_ref)
    input = torch.randn(2)

    pred: t.List[torch.Tensor] = worker.execute(model_ref, [input])

    assert pred


def test_execute_missing_model(persist_model_file: pathlib.Path) -> None:
    """Verify that a executing a model with an invalid key fails cleanly"""

    # todo: consider moving to file specific to key tests

    # use key that references an un-set model value
    key = mli.MemoryKey("test-key", b"")

    worker = mli.DefaultTorchWorker
    model_ref = mli.MachineLearningModelRef(worker.backend(), key)

    with pytest.raises(ValueError) as ex:
        model_ref.model()

    assert "empty value" in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_transform_output() -> None:
    """Verify that the default output transform operation is a no-op copy"""
    rows, cols = 1, 4
    num_values = 7
    inputs = [torch.randn((rows, cols)) for _ in range(num_values)]
    exp_outputs = [torch.Tensor(tensor) for tensor in inputs]

    worker = mli.DefaultTorchWorker
    transformed: t.Collection[torch.Tensor] = worker.transform_output(inputs)

    assert len(transformed) == num_values

    for output, expected in zip(transformed, exp_outputs):
        assert output.shape == expected.shape
        assert output.equal(expected)

    # verify a copy was made
    original: torch.Tensor = inputs[0]
    transformed[0] = 2 * transformed[0]

    assert transformed[0].equal(2 * original)
