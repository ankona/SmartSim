import io
import logging
import multiprocessing as mp
import pathlib
import pickle
import time
import typing as t
import uuid
from abc import ABC, abstractmethod

import torch

import smartsim.error as sse
from smartsim.log import get_logger

if t.TYPE_CHECKING:
    import dragon.channels as dch

logger = get_logger(__name__)


class DragonDict:
    """Mock implementation of a dragon dictionary"""

    def __init__(self) -> None:
        """Initialize the mock DragonDict instance"""
        self._storage: t.Dict[bytes, t.Any] = {}

    def __getitem__(self, key: bytes) -> t.Any:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        return self._storage[key]

    def __setitem__(self, key: bytes, value: t.Any) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        self._storage[key] = value

    def __contains__(self, key: bytes) -> bool:
        """Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        return key in self._storage


class FeatureStore(ABC):
    """Abstract base class providing the common interface for retrieving
    values from a feature store implementation"""

    @abstractmethod
    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""

    @abstractmethod
    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""


class MemoryFeatureStore(FeatureStore):
    """A feature store with values persisted only in local memory"""

    def __init__(self) -> None:
        """Initialize the MemoryFeatureStore instance"""
        self._storage: t.Dict[str, bytes] = {}

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        if key not in self._storage:
            raise sse.SmartSimError(f"{key} not found in feature store")
        return self._storage[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        return key in self._storage


class DragonFeatureStore(FeatureStore):
    """A feature store backed by a dragon distributed dictionary"""

    def __init__(self, storage: DragonDict) -> None:
        """Initialize the DragonFeatureStore instance"""
        self._storage = storage

    def __getitem__(self, key: str) -> t.Any:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        key_ = key.encode("utf-8")
        if key_ not in self._storage:
            raise sse.SmartSimError(f"{key} not found in feature store")
        return self._storage[key_]

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        key_ = key.encode("utf-8")
        self._storage[key_] = value

    def __contains__(self, key: t.Union[str, bytes]) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        if isinstance(key, str):
            key = key.encode("utf-8")
        return key in self._storage


class CommChannel(ABC):
    """Base class for abstracting a message passing mechanism"""

    @abstractmethod
    def send(self, value: bytes) -> None:
        """Send a message throuh the underlying communication channel
        :param value: The value to send"""

    @classmethod
    @abstractmethod
    def find(cls, key: bytes) -> "CommChannel":
        """Find a channel given its serialized key
        :param key: The unique descriptor of a communications channel"""
        raise NotImplementedError()


class DragonCommChannel(CommChannel):
    """Passes messages by writing to a Dragon channel"""

    def __init__(self, channel: "dch.Channel") -> None:
        """Initialize the DragonCommChannel instance"""
        self._channel = channel

    def send(self, value: bytes) -> None:
        """Send a message throuh the underlying communication channel
        :param value: The value to send"""
        self._channel.send_bytes(value)


class InferenceRequest:
    """Temporary model of an inference request"""

    def __init__(
        self,
        model_key: t.Optional[str] = None,
        callback: t.Optional[CommChannel] = None,
        raw_inputs: t.Optional[t.List[bytes]] = None,
        input_keys: t.Optional[t.List[str]] = None,
        output_keys: t.Optional[t.List[str]] = None,
        raw_model: t.Optional[bytes] = None,
        batch_size: int = 0,
    ):
        """Initialize the InferenceRequest"""
        self.model_key = model_key
        self.raw_model = raw_model
        self.callback = callback
        self.raw_inputs = raw_inputs
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.batch_size = batch_size


class InferenceReply:
    """Temporary model of a reply to an inference request"""

    def __init__(
        self,
        outputs: t.Optional[t.Collection[bytes]] = None,
        output_keys: t.Optional[t.Collection[str]] = None,
    ) -> None:
        """Initialize the InferenceReply"""
        self.outputs: t.Collection[bytes] = outputs or []
        self.output_keys: t.Collection[t.Optional[str]] = output_keys or []


class ModelLoadResult:
    """A wrapper around a loaded model"""

    def __init__(self, model: t.Any) -> None:
        """Initialize the ModelLoadResult"""
        self.model = model


class InputTransformResult:
    """A wrapper around a transformed input"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the InputTransformResult"""
        self.transformed = result


class ExecuteResult:
    """A wrapper around inference results"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the ExecuteResult"""
        self.predictions = result


class InputFetchResult:
    """A wrapper around fetched inputs"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the InputFetchResult"""
        self.inputs = result


class OutputTransformResult:
    """A wrapper around inference results transformed for transmission"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the OutputTransformResult"""
        self.outputs = result


class BatchResult:
    """A wrapper around batched inputs"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the BatchResult"""
        self.batch = result


class FetchModelResult:
    """A wrapper around raw fetched models"""

    def __init__(self, result: bytes) -> None:
        """Initialize the BatchResult"""
        self.model_bytes = result


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types"""

    @staticmethod
    def fetch_model(
        request: InferenceRequest, feature_store: FeatureStore
    ) -> FetchModelResult:
        """Given a resource key, retrieve the raw model from a feature store
        :param request: The request that triggered the pipeline
        :param feature_store: The feature store used for persistence
        :return: Raw bytes of the model"""
        if request.raw_model:
            # Should we cache model in the feature store?
            # model_key = hash(request.raw_model)
            # feature_store[model_key] = request.raw_model
            # short-circuit and return the directly supplied model
            return FetchModelResult(request.raw_model)

        if not request.model_key:
            raise sse.SmartSimError(
                "Key must be provided to retrieve model from feature store"
            )

        try:
            raw_bytes = feature_store[request.model_key]
            return FetchModelResult(raw_bytes)
        except FileNotFoundError as ex:
            logger.exception(ex)
            raise sse.SmartSimError(
                f"Model could not be retrieved with key {request.model_key}"
            ) from ex

    @staticmethod
    def fetch_inputs(
        request: InferenceRequest, feature_store: FeatureStore
    ) -> InputFetchResult:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata
        :param request: The request that triggered the pipeline
        :param feature_store: The feature store used for persistence
        :return: the fetched input"""
        if request.input_keys:
            data: t.List[bytes] = []
            for input_ in request.input_keys:
                try:
                    tensor_bytes = feature_store[input_]
                    data.append(tensor_bytes)
                except KeyError as ex:
                    logger.exception(ex)
                    raise sse.SmartSimError(
                        f"Model could not be retrieved with key {input_}"
                    ) from ex
            return InputFetchResult(data)

        if request.raw_inputs:
            return InputFetchResult(request.raw_inputs)

        raise ValueError("No input source")

    @staticmethod
    def batch_requests(
        request: InferenceRequest, transform_result: InputTransformResult
    ) -> BatchResult:
        """Create a batch of requests. Return the batch when batch_size datum have been
        collected or a configured batch duration has elapsed.
        :param request: The request that triggered the pipeline
        :param transform_result: Transformed inputs ready for batching
        :return: `None` if batch size has not been reached and timeout not exceeded."""
        if transform_result is not None or request.batch_size:
            raise NotImplementedError("Batching is not yet supported")
        return BatchResult(None)

    @staticmethod
    def place_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
        feature_store: FeatureStore,
    ) -> t.Collection[t.Optional[str]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store
        :param request: The request that triggered the pipeline
        :param execute_result: Results from inference
        :param feature_store: The feature store used for persistence"""
        keys: t.List[t.Optional[str]] = []
        # need to decide how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.

        for k, v in zip(request.output_keys, execute_result.predictions):
            feature_store[k] = v
            keys.append(k)

        return keys


class MachineLearningWorkerBase(MachineLearningWorkerCore, ABC):
    """Abstrct base class providing contract for a machine learning
    worker implementation."""

    @staticmethod
    @abstractmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend
        :param data_blob: inference request as a byte-serialized blob
        :return: InferenceRequest deserialized from the input"""

    @staticmethod
    @abstractmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult
    ) -> ModelLoadResult:
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory
        :param request: The request that triggered the pipeline
        :return: ModelLoadResult wrapping the model loaded for the request"""

    @staticmethod
    @abstractmethod
    def transform_input(
        request: InferenceRequest, fetch_result: InputFetchResult
    ) -> InputTransformResult:
        """Given a collection of data, perform a transformation on the data
        :param request: The request that triggered the pipeline
        :param fetch_result: Raw output from fetching inputs out of a feature store
        :return: The transformed inputs wrapped in a InputTransformResult"""

    @staticmethod
    @abstractmethod
    def execute(
        request: InferenceRequest,
        load_result: ModelLoadResult,
        transform_result: InputTransformResult,
    ) -> ExecuteResult:
        """Execute an ML model on inputs transformed for use by the model
        :param request: The request that triggered the pipeline
        :param load_result: The result of loading the model onto device memory
        :param transform_result: The result of transforming inputs for model consumption
        :return: The result of inference wrapped in an ExecuteResult"""

    @staticmethod
    @abstractmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
    ) -> OutputTransformResult:
        """Given inference results, perform transformations required to
        transmit results to the requestor.
        :param request: The request that triggered the pipeline
        :param execute_result: The result of inference wrapped in an ExecuteResult
        :return:"""

    @staticmethod
    @abstractmethod
    def serialize_reply(reply: InferenceReply) -> bytes:
        """Given an output, serialize to bytes for transport
        :param reply: The result of the inference pipeline
        :return: a byte-serialized version of the reply"""


class SampleTorchWorker(MachineLearningWorkerBase):
    """A minimum implementation of a worker that executes a PyTorch model"""

    @staticmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        request: InferenceRequest = pickle.loads(data_blob)
        return request

    @staticmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult
    ) -> ModelLoadResult:
        model_bytes = fetch_result.model_bytes or request.raw_model
        if not model_bytes:
            raise ValueError("Unable to load model without reference object")

        model: torch.nn.Module = torch.load(io.BytesIO(model_bytes))
        result = ModelLoadResult(model)
        return result

    @staticmethod
    def transform_input(
        request: InferenceRequest, fetch_result: InputFetchResult
    ) -> InputTransformResult:
        result = [torch.load(io.BytesIO(item)) for item in fetch_result.inputs]
        return InputTransformResult(result)
        # return data # note: this fails copy test!

    @staticmethod
    def execute(
        request: InferenceRequest,
        load_result: ModelLoadResult,
        transform_result: InputTransformResult,
    ) -> ExecuteResult:
        """Execute an ML model on the given inputs"""
        if not load_result.model:
            raise sse.SmartSimError("Model must be loaded to execute")

        model = load_result.model
        results = [model(tensor) for tensor in transform_result.transformed]

        execute_result = ExecuteResult(results)
        return execute_result

    # todo: ask team if we should always do in-place to avoid copying everything
    @staticmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
    ) -> OutputTransformResult:
        transformed = [item.clone() for item in execute_result.predictions]
        return OutputTransformResult(transformed)

    @staticmethod
    def serialize_reply(reply: InferenceReply) -> bytes:
        return pickle.dumps(reply)


class ServiceHost(ABC):
    """Base contract for standalone entrypoint scripts. Defines API for entrypoint
    behaviors (event loop, automatic shutdown, cooldown) as well as simple
    hooks for status changes"""

    def __init__(self, as_service: bool = False, cooldown: int = 0) -> None:
        """Initialize the ServiceHost
        :param as_service: Determines if the host will run until shutdown criteria
        are met or as a run-once instance
        :param cooldown: Period of time to allow service to run before automatic
        shutdown, in seconds. A non-zero, positive integer."""
        self._as_service = as_service
        """If the service should run until shutdown function returns True"""
        self._cooldown = cooldown
        """Duration of a cooldown period between requests to the service
        before shutdown"""

    @abstractmethod
    def _on_iteration(self, timestamp: int) -> None:
        """The user-defined event handler. Executed repeatedly until shutdown
        conditions are satisfied and cooldown is elapsed.
        :param timestamp: the timestamp at the start of the event loop iteration
        """

    @abstractmethod
    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""

    def _on_start(self) -> None:
        """Empty hook method for use by subclasses. Called on initial entry into
        ServiceHost `execute` event loop before `_on_iteration` is invoked."""
        logger.debug(f"Starting {self.__class__.__name__}")

    def _on_shutdown(self) -> None:
        """Empty hook method for use by subclasses. Called immediately after exiting
        the main event loop during automatic shutdown."""
        logger.debug(f"Shutting down {self.__class__.__name__}")

    def _on_cooldown(self) -> None:
        """Empty hook method for use by subclasses. Called on every event loop
        iteration immediately upon exceeding the cooldown period"""
        logger.debug(f"Cooldown exceeded by {self.__class__.__name__}")

    def execute(self) -> None:
        """The main event loop of a service host. Evaluates shutdown criteria and
        combines with a cooldown period to allow automatic service termination.
        Responsible for executing calls to subclass implementation of `_on_iteration`"""
        self._on_start()

        start_ts = time.time_ns()
        last_ts = start_ts
        running = True
        elapsed_cooldown = 0
        nanosecond_scale_factor = 1000000000
        cooldown_ns = self._cooldown * nanosecond_scale_factor

        # if we're run-once, use cooldown to short circuit
        if not self._as_service:
            self._cooldown = 1
            last_ts = start_ts - (cooldown_ns * 2)

        while running:
            self._on_iteration(start_ts)

            eligible_to_quit = self._can_shutdown()

            if self._cooldown and not eligible_to_quit:
                # reset timer any time cooldown is interrupted
                elapsed_cooldown = 0

            # allow service to shutdown if no cooldown period applies...
            running = not eligible_to_quit

            # ... but verify we don't have remaining cooldown time
            if self._cooldown:
                elapsed_cooldown += start_ts - last_ts
                remaining = cooldown_ns - elapsed_cooldown
                running = remaining > 0

                rem_in_s = remaining / nanosecond_scale_factor

                if not running:
                    cd_in_s = cooldown_ns / nanosecond_scale_factor
                    logger.info(f"cooldown {cd_in_s}s exceeded by {abs(rem_in_s):.2f}s")
                    self._on_cooldown()
                    continue

                logger.debug(f"cooldown remaining {abs(rem_in_s):.2f}s")

            last_ts = start_ts
            start_ts = time.time_ns()
            time.sleep(1)

        self._on_shutdown()


class WorkerManager(ServiceHost):
    """An implementation of a service managing distribution of tasks to
    machine learning workers"""

    def __init__(
        self,
        feature_store: FeatureStore,
        worker: MachineLearningWorkerBase,
        as_service: bool = False,
        cooldown: int = 0,
    ) -> None:
        """Initialize the WorkerManager
        :param feature_store: The persistence mechanism
        :param worker: A worker to manage
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down afer
        shutdown criteria are met"""
        super().__init__(as_service, cooldown)

        self._workers: t.Dict[
            str, "t.Tuple[MachineLearningWorkerBase, mp.Queue[bytes]]"
        ] = {}
        """a collection of workers the manager is controlling"""
        self._upstream_queue: t.Optional[mp.Queue[bytes]] = None
        """the queue the manager monitors for new tasks"""
        self._feature_store: FeatureStore = feature_store
        """a feature store to retrieve models from"""
        self._worker = worker
        """The ML Worker implementation"""

    @property
    def upstream_queue(self) -> "t.Optional[mp.Queue[bytes]]":
        """Return the queue used by the worker manager to receive new work"""
        return self._upstream_queue

    @upstream_queue.setter
    def upstream_queue(self, value: "mp.Queue[bytes]") -> None:
        """Set/update the queue used by the worker manager to receive new work"""
        self._upstream_queue = value

    def _on_iteration(self, timestamp: int) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""
        logger.debug(f"{timestamp} executing worker manager pipeline")

        if self.upstream_queue is None:
            logger.warning("No queue to check for tasks")
            return

        msg: bytes = self.upstream_queue.get()

        request = self._worker.deserialize(msg)
        fetch_model_result = self._worker.fetch_model(request, self._feature_store)
        model_result = self._worker.load_model(request, fetch_model_result)
        fetch_input_result = self._worker.fetch_inputs(
            request,
            self._feature_store,
        )  # we don't know if they'lll fetch in some weird way
        # they potentially need access to custom attributes
        # we don't know what the response really is... i have it as bytes
        # but we just want to advertise that the contract states "the output
        # will be the input to transform_input... "

        transform_result = self._worker.transform_input(request, fetch_input_result)

        # batch: t.Collection[_Datum] = transform_result.transformed_input
        # if self._batch_size:
        #     batch = self._worker.batch_requests(transform_result, self._batch_size)

        # todo: what do we return here? tensors? Datum? bytes?
        results = self._worker.execute(request, model_result, transform_result)

        reply = InferenceReply()

        # only place into feature store if keys are provided
        if request.output_keys:
            output_keys = self._worker.place_output(
                request, results, self._feature_store
            )
            reply.output_keys = output_keys
        else:
            reply.outputs = results.predictions

        serialized_output = self._worker.serialize_reply(reply)

        callback_channel = request.callback
        if callback_channel:
            callback_channel.send(serialized_output)

    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""
        return bool(self._workers)

    def add_worker(
        self, worker: MachineLearningWorkerBase, work_queue: "mp.Queue[bytes]"
    ) -> None:
        """Add a worker instance to the collection managed by the WorkerManager"""
        self._workers[str(uuid.uuid4())] = (worker, work_queue)


def mock_work(worker_manager_queue: "mp.Queue[bytes]") -> None:
    """Mock event producer for triggering the inference pipeline"""
    # todo: move to unit tests
    while True:
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        timestamp = time.time_ns()
        test_dir = "/lus/bnchlu1/mcbridch/code/ss/tests/test_output/brainstorm"
        test_path = pathlib.Path(test_dir)

        mock_channel = test_path / f"brainstorm-{timestamp}.txt"
        mock_model = test_path / "brainstorm.pt"

        test_path.mkdir(parents=True, exist_ok=True)
        mock_channel.touch()
        mock_model.touch()

        msg = f"PyTorch:{mock_model}:MockInputToReplace:{mock_channel}"
        worker_manager_queue.put(msg.encode("utf-8"))


if __name__ == "__main__":
    logging.basicConfig(filename="workermanager.log")
    # queue for communicating to the worker manager. used to
    # simulate messages "from the application"
    upstream_queue: "mp.Queue[bytes]" = mp.Queue()

    # downstream_queue = mp.Queue()  # the queue to forward messages to a given worker

    # torch_worker = TorchWorker(downstream_queue)

    # dict_fs = DictFeatureStore()

    # worker_manager = WorkerManager(dict_fs, as_service=True, cooldown=10)
    # # configure what the manager listens to
    # worker_manager.upstream_queue = upstream_queue
    # # # and configure a worker ... moving...
    # # will dynamically add a worker in the manager based on input msg backend
    # # worker_manager.add_worker(torch_worker, downstream_queue)

    # # create a pretend to populate the queues
    msg_pump = mp.Process(target=mock_work, args=(upstream_queue,))
    msg_pump.start()

    # # create a process to process commands
    # process = mp.Process(target=worker_manager.execute, args=(time.time_ns(),))
    # process.start()
    # process.join()

    # msg_pump.kill()
    # logger.info(f"{DefaultTorchWorker.backend()=}")
