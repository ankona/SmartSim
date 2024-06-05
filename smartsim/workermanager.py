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

_Datum = t.Union[torch.Tensor]
logger = get_logger(__name__)


class DragonDict:
    """Mock implementation of a dragon dictionary"""

    def __init__(self) -> None:
        """Initialize the mock DragonDict instance"""
        self._storage: t.Dict[bytes, t.Any] = {}

    def __getitem__(self, key: bytes) -> t.Any:
        """Retrieve an item using key"""
        return self._storage[key]

    def __setitem__(self, key: bytes, item: t.Any) -> None:
        """Assign a value using key"""
        self._storage[key] = item

    def __contains__(self, key: bytes) -> bool:
        """Return `True` if the key is found, `False` otherwise"""
        return key in self._storage


class FeatureStore(ABC):
    @abstractmethod
    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key"""

    @abstractmethod
    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key"""

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Return `True` if the key is found, `False` otherwise"""


class MemoryFeatureStore(FeatureStore):
    def __init__(self) -> None:
        """Initialize the MemoryFeatureStore instance"""
        self._storage: t.Dict[str, bytes] = {}  # defaultdict(lambda: None)

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key"""
        if key not in self._storage:
            raise sse.SmartSimError(f"{key} not found in feature store")
        return self._storage[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key"""
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        """Return `True` if the key is found, `False` otherwise"""
        return key in self._storage


class FileSystemFeatureStore(FeatureStore):
    def __init__(self, storage_dir: t.Optional[pathlib.Path] = None) -> None:
        """Initialize the FileSystemFeatureStore instance"""
        self._storage_dir = storage_dir

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key"""
        path = self._key_path(key)
        if not path.exists():
            raise sse.SmartSimError(f"{path} not found in feature store")
        return path.read_bytes()

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key"""
        path = self._key_path(key)
        path.write_bytes(value)

    def __contains__(self, key: str) -> bool:
        """Return `True` if the key is found, `False` otherwise"""
        path = self._key_path(key)
        return path.exists()

    def _key_path(self, key: str) -> pathlib.Path:
        """Given a key, return a path that is optionally combined with a base
        directory used by the FileSystemFeatureStore."""
        if self._storage_dir:
            return self._storage_dir / key

        return pathlib.Path(key)


class DragonFeatureStore(FeatureStore):
    def __init__(self, storage: DragonDict) -> None:
        """Initialize the DragonFeatureStore instance"""
        self._storage = storage

    def __getitem__(self, key: str) -> t.Any:
        """Retrieve an item using key"""
        key_ = key.encode("utf-8")
        if key_ not in self._storage:
            raise sse.SmartSimError(f"{key} not found in feature store")
        return self._storage[key_]

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key"""
        key_ = key.encode("utf-8")
        self._storage[key_] = value

    def __contains__(self, key: t.Union[str, bytes]) -> bool:
        """Return `True` if the key is found, `False` otherwise"""
        if isinstance(key, str):
            key = key.encode("utf-8")
        return key in self._storage


class CommChannel(ABC):
    """Base class for abstracting a message passing mechanism"""

    @abstractmethod
    def send(self, value: bytes) -> None:
        """Send the supplied value through the underlying channel as a message"""

    @classmethod
    @abstractmethod
    def find(cls, key: bytes) -> "CommChannel":
        """Find a channel given its serialized key"""
        raise NotImplementedError()


class DragonCommChannel(CommChannel):
    """Passes messages by writing to a Dragon channel"""

    def __init__(self, channel: "dch.Channel") -> None:
        """Initialize the DragonCommChannel instance"""
        self._channel = channel

    def send(self, value: bytes) -> None:
        """Write the supplied value to a Dragon channel"""
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
    def __init__(self, model: t.Any) -> None:
        self.model = model


class InputTransformResult:
    def __init__(self, result: t.Any) -> None:
        self.transformed_input = result


class ExecuteResult:
    def __init__(self, result: t.Any) -> None:
        self.predictions = result


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types"""

    @staticmethod
    def fetch_model(request: InferenceRequest, feature_store: FeatureStore) -> bytes:
        """Given a ResourceKey, identify the physical location and model metadata"""
        key = request.model_key
        if not key:
            raise sse.SmartSimError(
                "Key must be provided to retrieve model from feature store"
            )

        try:
            return feature_store[key]
        except FileNotFoundError as ex:
            logger.exception(ex)
            raise sse.SmartSimError(
                f"Model could not be retrieved with key {key}"
            ) from ex

    @staticmethod
    def fetch_inputs(
        request: InferenceRequest, feature_store: FeatureStore
    ) -> t.Collection[bytes]:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata

        :param inputs: Collection of keys identifying values in a feature store
        :param feature_store: The feature store in use by the worker manager

        :return: Raw bytes identified by the given keys when found, otherwise `None`
        """
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
            return data

        if request.raw_inputs:
            return request.raw_inputs

        raise ValueError("No input source")

    @staticmethod
    def batch_requests(
        request: InferenceRequest, transform_result: InputTransformResult
    ) -> t.Collection[_Datum]:
        """Create a batch of requests. Return the batch when batch_size datum have been
        collected or a configured batch duration has elapsed.

        :param data: Collection of input messages that may be added to the current batch
        :param batch_size: The maximum allowed batch size

        :return: `None` if batch size has not been reached and timeout not exceeded."""
        if transform_result is not None or request.batch_size:
            raise NotImplementedError("Batching is not yet supported")
        return []

    @staticmethod
    def place_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
        feature_store: FeatureStore,
        # need to know how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.
    ) -> t.Collection[t.Optional[str]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store"""
        keys: t.List[t.Optional[str]] = []

        for k, v in zip(request.output_keys, execute_result.predictions):
            feature_store[k] = v
            keys.append(k)

        return keys


class MachineLearningWorkerBase(MachineLearningWorkerCore, ABC):
    @staticmethod
    @abstractmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend"""

    @staticmethod
    @abstractmethod
    def load_model(request: InferenceRequest) -> ModelLoadResult:
        # model: MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory"""
        # invoke separate API functions to put the model on GPU/accelerator (if exists)

    @staticmethod
    @abstractmethod
    def transform_input(
        request: InferenceRequest,
        data: t.Collection[bytes],
    ) -> InputTransformResult:
        """Given a collection of data, perform a transformation on the data"""

    @staticmethod
    @abstractmethod
    def execute(
        request: InferenceRequest,
        load_result: ModelLoadResult,
        transform_result: InputTransformResult,
    ) -> ExecuteResult:
        """Execute an ML model on the given inputs"""

    @staticmethod
    @abstractmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
    ) -> t.Collection[_Datum]:
        """Given a collection of data, perform a transformation on the data"""

    @staticmethod
    @abstractmethod
    def serialize_reply(reply: InferenceReply) -> bytes:
        """Given an output, serialize to bytes for transport"""


class DefaultTorchWorker(MachineLearningWorkerBase):
    @staticmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        """Given a byte-serialized request, convert the bytes
        to a proper representation for use by the ML backend"""
        # TODO: note - this is temporary (using pickle) until a serializer is
        # created and we replace it...
        request: InferenceRequest = pickle.loads(data_blob)
        return request

    @staticmethod
    def load_model(request: InferenceRequest) -> ModelLoadResult:
        # MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory"""
        if not request.raw_model:
            raise ValueError("Unable to load model without reference object")

        # invoke separate API functions to put the model on GPU/accelerator (if exists)
        raw_bytes = request.raw_model or b""  # todo: ???
        model_bytes = io.BytesIO(raw_bytes)
        model: torch.nn.Module = torch.load(model_bytes)
        result = ModelLoadResult(model)
        return result

    @staticmethod
    def transform_input(
        request: InferenceRequest,
        data: t.Collection[bytes],
    ) -> InputTransformResult:
        """Given a collection of data, perform a no-op, copy-only transform"""
        result = [torch.load(io.BytesIO(item)) for item in data]
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
        results = [model(tensor) for tensor in transform_result.transformed_input]

        execute_result = ExecuteResult(results)
        return execute_result

    # todo: ask team if we should always do in-place to avoid copying everything
    @staticmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
        # TODO: ask Al about assumption that "if i put in tensors, i will get out
        # tensors. my generic probably fails here."
    ) -> t.Collection[_Datum]:
        """Given a collection of data, perform a no-op, copy-only transform"""
        return [item.clone() for item in execute_result.predictions]

    @staticmethod
    def serialize_reply(reply: InferenceReply) -> bytes:
        """Given an output, serialize to bytes for transport"""
        return pickle.dumps(reply)


class ServiceHost(ABC):
    """Nice place to have some default entrypoint junk (args, event
    loop, cooldown, etc)"""

    def __init__(self, as_service: bool = False, cooldown: int = 0) -> None:
        self._as_service = as_service
        """If the service should run until shutdown function returns True"""
        self._cooldown = cooldown
        """Duration of a cooldown period between requests to the service
        before shutdown"""

    @abstractmethod
    def _on_iteration(self, timestamp: int) -> None:
        """The user-defined event handler. Executed repeatedly until shutdown
        conditions are satisfied and cooldown is elapsed.
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
    def __init__(
        self,
        feature_store: FeatureStore,
        worker: MachineLearningWorkerBase,
        as_service: bool = False,
        cooldown: int = 0,
        batch_size: int = 0,
    ) -> None:
        """Initialize the WorkerManager"""
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
        self._batch_size = batch_size
        """The number of inputs to batch for execution."""

    @property
    def upstream_queue(self) -> "t.Optional[mp.Queue[bytes]]":
        """Return the queue used by the worker manager to receive new work"""
        return self._upstream_queue

    @upstream_queue.setter
    def upstream_queue(self, value: "mp.Queue[bytes]") -> None:
        """Set/update the queue used by the worker manager to receive new work"""
        self._upstream_queue = value

    @property
    def batch_size(self) -> int:
        """Returns the maximum size of a batch to be sent to a worker"""
        return self._batch_size

    def _on_iteration(self, timestamp: int) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""
        logger.debug(f"{timestamp} executing worker manager pipeline")

        if self.upstream_queue is None:
            logger.warning("No queue to check for tasks")
            return

        msg: bytes = self.upstream_queue.get()

        request = self._worker.deserialize(msg)

        # request_ctx = RequestContext()
        # self._worker.fetch_model(request_ctx)

        # m = feature_store[key]
        # split load model & fetch model
        # load model is SPECIFICALLY load onto GPU, not from feature_store
        model_result = self._worker.load_model(request)

        fetched_inputs = self._worker.fetch_inputs(
            request,
            # request.input_keys, self._feature_store, request
            self._feature_store,
        )  # we don't know if they'lll fetch in some weird way
        # they potentially need access to custom attributes
        # we don't know what the response really is... i have it as bytes
        # but we just want to advertise that the contract states "the output
        # will be the input to transform_input... "

        # a_resp = do_a()
        # b_resp = do_b(a_resp)
        # c_resp = do_c(b_resp)

        # with Pool(4) as p:
        # p.(worker.transform_input, fetch_inputs)
        # p.start()
        transform_result = self._worker.transform_input(request, fetched_inputs)

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


# p = mp.Process(target=lambda: 1, args=(1, 2, 3))
# p.start()


# class ProcessingContext:
#     def __init__(self) -> None:
#         deserialize_out: t.Any
#         model: t.Any
#         inputs: t.List[t.Any]
#         transformed_inputs: t.List[t.Any]
#         batches: t.Any  # : ??? <--- how does the next stage know when a batch is new)
#         results: t.List[t.Any]
#         transformed_results: t.List[t.Any]
#         persisted_keys: t.List[t.Any]
#         serialized_results: t.List[t.Any]


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
