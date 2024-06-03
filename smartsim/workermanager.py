import io
import multiprocessing as mp
import pathlib
import pickle
import time
import typing as t
from abc import ABC, abstractmethod

import torch

import smartsim.error as sse

if t.TYPE_CHECKING:
    import dragon.channels as dch

_Datum = t.Union[torch.Tensor]


class DragonDict:
    """mock out the dragon dict..."""

    def __init__(self) -> None:
        self._storage: t.Dict[bytes, t.Any] = {}

    def __getitem__(self, key: bytes) -> t.Any:
        return self._storage[key]

    def __setitem__(self, key: bytes, item: t.Any) -> None:
        self._storage[key] = item

    def __contains__(self, key: bytes) -> bool:
        return key in self._storage


class ResourceKey(ABC):
    """Uniquely identify the resource by location"""

    def __init__(self, key: str) -> None:
        self._key = key

    @property
    def key(self) -> str:
        return self._key

    @abstractmethod
    def retrieve(self) -> bytes: ...

    @abstractmethod
    def put(self, value: bytes) -> None: ...


class FeatureStore(ABC):
    @abstractmethod
    def __getitem__(self, key: str) -> bytes: ...

    @abstractmethod
    def __setitem__(self, key: str, value: bytes) -> None: ...

    @abstractmethod
    def __contains__(self, key: str) -> bool: ...

    def get_key(self, key: str) -> t.Optional[ResourceKey]:
        if key in self:
            return FeatureStoreKey(key, self)
        return None


class DictFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self._storage: t.Dict[str, bytes] = {}  # defaultdict(lambda: None)

    def __getitem__(self, key: str) -> bytes:
        return self._storage[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._storage


class DragonFeatureStore(FeatureStore):
    def __init__(self, storage: DragonDict) -> None:
        self._storage = storage

    def __getitem__(self, key: str) -> t.Any:
        key_ = key.encode("utf-8")
        return self._storage[key_]

    def __setitem__(self, key: str, value: bytes) -> None:
        key_ = key.encode("utf-8")
        self._storage[key_] = value

    def __contains__(self, key: t.Union[str, bytes]) -> bool:
        if isinstance(key, str):
            key = key.encode("utf-8")
        return key in self._storage


class FeatureStoreKey(ResourceKey):
    def __init__(self, key: str, feature_store: FeatureStore):
        super().__init__(key)
        self._feature_store = feature_store

    def retrieve(self) -> bytes:
        if self._key not in self._feature_store:
            raise KeyError(f"`{self._key}` not found in feature store")
        return self._feature_store[self._key]

    def put(self, value: bytes) -> None:
        self._feature_store[self._key] = value


class FileSystemKey(ResourceKey):
    def __init__(self, path: pathlib.Path):
        super().__init__(path.absolute().as_posix())

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self._key)

    def retrieve(self) -> bytes:
        if not self.path.exists():
            raise KeyError(f"Invalid FileSystemKey; `{self.path}` was not found")

        return self.path.read_bytes()

    def put(self, value: bytes) -> None:
        with self.path as write_to:
            write_to.write_bytes(value)


class MemoryKey(ResourceKey):
    def __init__(self, key: str, value: bytes):
        super().__init__(key)
        self._value = value

    def retrieve(self) -> bytes:
        if not self._value:
            raise ValueError("MemoryKey references empty value")
        return self._value

    def put(self, value: bytes) -> None:
        self._value = value


class MachineLearningModelRef:
    def __init__(self, backend: str, key: t.Optional[ResourceKey] = None) -> None:
        self._backend = backend
        self._key: t.Optional[ResourceKey] = key

    def model(self) -> bytes:
        if not self._key:
            raise ValueError("Key not set")
        return self._key.retrieve()

    @property
    def backend(self) -> str:
        return self._backend


class CommChannel(ABC):
    @abstractmethod
    def send(self, value: bytes) -> None: ...

    @classmethod
    @abstractmethod
    def find(cls, key: bytes) -> "CommChannel":
        """A way to find a channel with only a serialized key/descriptor"""
        raise NotImplementedError()


class FileCommChannel(CommChannel):
    def __init__(self, path: pathlib.Path) -> None:
        self._path: pathlib.Path = path

    def send(self, value: bytes) -> None:
        msg = f"Sending {value.decode('utf-8')} through file channel"
        print(msg)
        self._path.write_text(msg)

    @classmethod
    def find(cls, key: bytes) -> "CommChannel":
        """A way to find a channel with only a serialized key/descriptor"""
        path = pathlib.Path(key.decode("utf-8"))
        if not path.exists():
            path.touch()
        return FileCommChannel(path)


class DragonCommChannel(CommChannel):
    def __init__(self, channel: "dch.Channel") -> None:
        self._channel = channel

    def send(self, value: bytes) -> None:
        # msg = f"Sending {value.decode('utf-8')} through file channel"
        # self._path.write_text(msg)
        self._channel.send_bytes(value)


class InferenceRequest:
    def __init__(
        self,
        backend: t.Optional[str] = None,
        model: t.Optional[MachineLearningModelRef] = None,
        callback: t.Optional[CommChannel] = None,
        value: t.Optional[bytes] = None,
        input_keys: t.Optional[t.List[str]] = None,
        output_keys: t.Optional[t.List[str]] = None,
    ):
        self.backend = backend
        self.model = model
        self.callback = callback
        self.value = value
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []


class InferenceReply:
    def __init__(
        self,
        outputs: t.Optional[t.Collection[bytes]] = None,
        output_keys: t.Optional[t.Collection[str]] = None,
    ) -> None:
        self.outputs: t.Collection[bytes] = outputs or []
        self.output_keys: t.Collection[t.Optional[str]] = output_keys or []


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types"""

    @staticmethod
    def fetch_model(key: ResourceKey) -> bytes:
        """Given a ResourceKey, identify the physical location and model metadata"""
        try:
            return key.retrieve()
        except KeyError as ex:
            print(ex)  # todo: logger.error
            raise sse.SmartSimError(
                f"Model could not be retrieved with key {key.key}"
            ) from ex
        # value = feature_store[key.key]
        # return value

    @staticmethod
    def fetch_inputs(
        inputs: t.Collection[str], feature_store: FeatureStore
    ) -> t.Collection[bytes]:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata"""
        data: t.List[bytes] = []
        for input_ in inputs:
            try:
                tensor_bytes = feature_store[input_]
                data.append(tensor_bytes)
            except KeyError as ex:
                print(ex)
                raise sse.SmartSimError(
                    f"Model could not be retrieved with key {input_}"
                ) from ex
        return data

    @staticmethod
    def batch_requests(
        data: t.Collection[_Datum], batch_size: int
    ) -> t.Collection[_Datum]:
        """Create a batch of requests. Return the batch when batch_size datum have been
        collected or a configured batch duration has elapsed.

        Returns `None` if batch size has not been reached and timeout not exceeded."""
        if data or batch_size:
            raise NotImplementedError("Batching is not yet supported")
        return []

    # todo: place_output is so awkward... why should client need to pass key type?
    # that should be decided by the feature store. fix it.

    @staticmethod
    def place_output(
        raw_keys: t.Collection[str],
        data: t.Collection[bytes],
        feature_store: FeatureStore,
        # need to know how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.
    ) -> t.Collection[t.Optional[ResourceKey]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store"""
        keys: t.List[t.Optional[ResourceKey]] = []

        for k, v in zip(raw_keys, data):
            feature_store[k] = v
            keys.append(feature_store.get_key(k))

        return keys


class MachineLearningWorkerBase(MachineLearningWorkerCore, ABC):
    @staticmethod
    @abstractmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend"""

    @staticmethod
    @abstractmethod
    def load_model(model_ref: t.Optional[MachineLearningModelRef]) -> t.Any:
        # model: MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory"""
        # invoke separate API functions to put the model on GPU/accelerator (if exists)

    @staticmethod
    @abstractmethod
    def transform_input(
        data: t.Collection[bytes],
    ) -> t.Collection[_Datum]:
        """Given a collection of data, perform a transformation on the data"""

    @staticmethod
    @abstractmethod
    def execute(
        model_ref: MachineLearningModelRef, data: t.Collection[_Datum]
    ) -> t.Collection[bytes]:
        """Execute an ML model on the given inputs"""

    @staticmethod
    @abstractmethod
    def transform_output(
        # TODO: ask Al about assumption that "if i put in tensors, i will get out
        # tensors. my generic probably fails here."
        data: t.Collection[_Datum],
    ) -> t.Collection[_Datum]:
        """Given a collection of data, perform a transformation on the data"""

    @staticmethod
    @abstractmethod
    def serialize_reply(reply: InferenceReply) -> bytes:
        """Given an output, serialize to bytes for transport"""

    @staticmethod
    @abstractmethod
    def backend() -> str: ...


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
    def load_model(model_ref: t.Optional[MachineLearningModelRef]) -> torch.nn.Module:
        # MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory"""
        if not model_ref:
            raise ValueError("Unable to load model without reference object")

        # invoke separate API functions to put the model on GPU/accelerator (if exists)
        raw_bytes = model_ref.model()
        model_bytes = io.BytesIO(raw_bytes)
        model: torch.nn.Module = torch.load(model_bytes)
        return model

    @staticmethod
    def transform_input(
        data: t.Collection[bytes],
    ) -> t.Collection[_Datum]:
        """Given a collection of data, perform a no-op, copy-only transform"""
        return [torch.load(io.BytesIO(item)) for item in data]
        # return data # note: this fails copy test!

    @staticmethod
    def execute(
        model_ref: MachineLearningModelRef, data: t.Collection[_Datum]
    ) -> t.Collection[bytes]:
        """Execute an ML model on the given inputs"""
        # todo: shouldn't be reloading model. need to pass model in instead?
        model = DefaultTorchWorker.load_model(model_ref)
        results = [model(tensor) for tensor in data]
        return results

    # todo: ask team if we should always do in-place to avoid copying everything
    @staticmethod
    def transform_output(
        data: t.Collection[_Datum],
        # TODO: ask Al about assumption that "if i put in tensors, i will get out
        # tensors. my generic probably fails here."
    ) -> t.Collection[_Datum]:
        """Given a collection of data, perform a no-op, copy-only transform"""
        return [item.clone() for item in data]

    @staticmethod
    def serialize_reply(reply: InferenceReply) -> bytes:
        """Given an output, serialize to bytes for transport"""
        return pickle.dumps(reply)

    @staticmethod
    def backend() -> str:
        return "PyTorch"


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
    def _on_iteration(self, timestamp: int) -> None: ...

    @abstractmethod
    def _can_shutdown(self) -> bool: ...

    def _on_start(self) -> None:
        print(f"Starting {self.__class__.__name__}")

    def _on_shutdown(self) -> None:
        print(f"Shutting down {self.__class__.__name__}")

    def _on_cooldown(self) -> None:
        print(f"Cooldown exceeded by {self.__class__.__name__}")

    def execute(self) -> None:  # , work_queue: mp.Queue) -> None:
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
                    print(f"cooldown {cd_in_s}s exceeded by {abs(rem_in_s):.2f}s")
                    self._on_cooldown()
                    continue

                print(f"cooldown remaining {abs(rem_in_s):.2f}s")

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
        return self._upstream_queue

    @upstream_queue.setter
    def upstream_queue(self, value: "mp.Queue[bytes]") -> None:
        self._upstream_queue = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _on_iteration(self, timestamp: int) -> None:
        print(f"{timestamp} executing worker manager pipeline")

        if self.upstream_queue is None:
            print("No queue to check for tasks")
            return

        msg: bytes = self.upstream_queue.get()

        request = self._worker.deserialize(msg)

        model = self._worker.load_model(request.model)
        fetched_inputs = self._worker.fetch_inputs(
            request.input_keys, self._feature_store
        )
        transformed_inputs = self._worker.transform_input(fetched_inputs)

        batch: t.Collection[_Datum] = transformed_inputs
        if self._batch_size:
            batch = self._worker.batch_requests(transformed_inputs, self._batch_size)

        # todo: what do we return here? tensors? Datum? bytes?
        results = self._worker.execute(model, batch)

        reply = InferenceReply()

        # only place into feature store if keys are provided
        if request.output_keys:
            output_keys = self._worker.place_output(
                request.output_keys, results, self._feature_store
            )

            keys: t.List[t.Optional[str]] = []
            for resource in output_keys:
                if resource is None:
                    keys.append(None)
                else:
                    keys.append(resource.key)

            reply.output_keys = keys
        else:
            reply.outputs = results

        serialized_output = self._worker.serialize_reply(reply)

        callback_channel = request.callback
        if callback_channel:
            callback_channel.send(serialized_output)

    def _can_shutdown(self) -> bool:
        return bool(self._workers)

    def add_worker(
        self, worker: MachineLearningWorkerBase, work_queue: "mp.Queue[bytes]"
    ) -> None:
        self._workers[worker.backend()] = (worker, work_queue)


def mock_work(worker_manager_queue: "mp.Queue[bytes]") -> None:
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

    upstream_queue: "mp.Queue[bytes]" = (
        mp.Queue()
    )  # the incoming messages from application
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
    print(f"{DefaultTorchWorker.backend()=}")
