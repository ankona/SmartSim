from collections import defaultdict
import enum
from functools import singledispatchmethod
from inspect import isfunction
from re import A
import typing as t
import pathlib
from abc import ABC, abstractmethod

import io
import time
from networkx import is_empty
import torch

# import tensorflow
# from tensorflow import io as tfio
import multiprocessing as mp


class DragonDict:
    """mock out the dragon dict..."""

    def __init__(self) -> None:
        self._storage = defaultdict(lambda: None)

    def __getitem__(self, key: t.Any) -> t.Any:
        return self._storage[key]

    def __setitem__(self, key: t.Any, item: t.Any) -> None:
        self._storage[key] = item


class ResourceKey(ABC):
    """Uniquely identify the resource by location"""

    def __init__(self, key: str) -> None:
        self._key = key

    #     self._uid: t.Optional[uuid.UUID] = None
    #     """A unique identifier for the resource"""
    #     self._loc: t.Optional[pathlib.Path] = None
    #     """A physical location where the resource can be retrieved"""
    #     self._key: t.Optional[str] = None
    #     """A key to reference the resource if it is already loaded in memory"""

    # @classmethod
    # def from_buffer(self, key: bytes, buffer: bytes) -> "ResourceKey":
    #     resource = ResourceKey(key)
    #     resource._value = buffer
    #     return resource

    @property
    # @abstractmethod
    def key(self) -> bytes:
        return self._key

    @abstractmethod
    def retrieve(self) -> bytes: ...

    @abstractmethod
    def put(self, value: bytes) -> None: ...


class FeatureStore(ABC):

    @abstractmethod
    def _get(self, key: bytes) -> bytes: ...

    @abstractmethod
    def _set(self, key: bytes, value: bytes) -> None: ...


class DictFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self._storage = defaultdict(lambda: None)

    def __getitem__(self, key: t.Any) -> t.Any:
        return self._get(key)

    def __setitem__(self, key: t.Any, item: t.Any) -> None:
        self._set(key, item)

    def _get(self, key: bytes) -> bytes:
        return self._storage[key]

    def _set(self, key: bytes, value: bytes) -> None:
        self._storage[key] = value


class DragonFeatureStore(FeatureStore):
    def __init__(self, storage: DragonDict) -> None:
        self._storage = storage

    def __getitem__(self, key: str) -> t.Any:
        return self._get(key.encode("utf-8"))

    def __setitem__(self, key: str, value: bytes) -> None:
        self._set(key.encode("utf-8"), value)

    def _get(self, key: bytes) -> bytes:
        return self._storage[key]

    def _set(self, key: bytes, value: bytes) -> None:
        self._storage[key] = value


class DragonFeatureStoreKey(ResourceKey):
    def __init__(self, key: bytes, feature_store: FeatureStore):
        super().__init__(key)
        self._feature_store = feature_store

    def retrieve(self) -> bytes:
        return self._feature_store[self._key]

    def put(self, value: bytes) -> None:
        self._feature_store[self._key] = value


class DictFeatureStoreKey(ResourceKey):
    def __init__(self, key: bytes, feature_store: FeatureStore):
        super().__init__(key)
        self._feature_store = feature_store

    @property
    def _decoded_key(self) -> str:
        return self._key.decode("utf-8")

    def retrieve(self) -> bytes:
        return self._feature_store[self._decoded_key]

    def put(self, value: bytes) -> None:
        self._feature_store[self._decoded_key] = value


class FeatureStoreKey(ResourceKey):
    def __init__(self, key: str, feature_store: FeatureStore):
        super().__init__(key)
        self._feature_store = feature_store

    def retrieve(self) -> bytes:
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
            raise FileNotFoundError(
                "Invalid FileSystemKey; requested file was not found"
            )

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
        return self._key.retrieve()

    @property
    def backend(self) -> str:
        return self._backend


_DatumType = t.TypeVar("_DatumType")


class Datum(t.Generic[_DatumType]):

    @property
    @abstractmethod
    def key(self) -> bytes: ...

    @property
    @abstractmethod
    def value(self) -> _DatumType: ...


class ResourceDatum(Datum[_DatumType]):
    def __init__(self, key: ResourceKey) -> None:
        self._key: ResourceKey = key

    def key(self) -> bytes:
        return self._key.key

    @property
    def value(self) -> Datum[_DatumType]:
        raw_bytes = self._key.retrieve()
        return self._transform_raw_bytes(raw_bytes)

    @abstractmethod
    def _transform_raw_bytes(self, raw_bytes: bytes) -> Datum[_DatumType]: ...


class TorchResource(ResourceDatum[torch.Tensor]):
    def __init__(self, key: ResourceKey, shape: t.Tuple[int]):
        super().__init__(key)
        self._shape = shape

    def _transform_raw_bytes(self, raw_bytes: bytes) -> torch.Tensor:
        storage = torch.Storage.from_buffer(raw_bytes)
        raw_tensor = torch.Tensor(storage)
        return raw_tensor.reshape(self._shape)


# class TensorflowResource(ResourceDatum[tensorflow.Tensor]):
#     def __init__(self, key: ResourceKey, shape: t.Tuple[int]):
#         super().__init__(key)
#         self._shape = shape

#     def _transform_raw_bytes(self, raw_bytes: bytes) -> tensorflow.Tensor:
#         raw_tensor: tensorflow.Tensor = tfio.decode_raw(
#             raw_bytes, tensorflow.float16
#         )  # <--- type must be from inputs...
#         return tensorflow.reshape(raw_tensor, self._shape)


class AggregationAction(str, enum.Enum):
    VSTACK = enum.auto()
    HSTACK = enum.auto()
    SUM = enum.auto()


class TransformAction(str, enum.Enum):
    TRANSPOSE = enum.auto()


# class StandardTransforms:
#     @staticmethod
#     def transform(action: TransformAction, data: t.Any) -> t.Any:
#         transformed = data
#         if action == TransformAction.TRANSPOSE:
#             # transpose the data...
#             transformed = StandardTransforms.transform(action, data)

#         return transformed

Transformer = t.Union[TransformAction, t.Callable[..., t.Any]]


class CommChannel(ABC):

    @abstractmethod
    def send(self, value: bytes) -> None: ...

    @classmethod
    @abstractmethod
    def find(cls, key: bytes) -> "CommChannel":
        """A way to find a channel with only a serialized key/descriptor"""
        raise NotImplemented()


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


if t.TYPE_CHECKING:
    import dragon.channels as dch


class DragonCommChannel(CommChannel):
    def __init__(self, channel: "dch.Channel") -> None:
        self._channel = channel

    def send(self, value: bytes) -> None:
        # msg = f"Sending {value.decode('utf-8')} through file channel"
        # self._path.write_text(msg)
        self._channel.send_bytes(value)


TorchAggregationFn = t.Callable[[t.Collection[torch.Tensor]], torch.Tensor]
TorchTransformFn = t.Callable[[torch.Tensor], torch.Tensor]


class TorchResourceAggregate:

    def __init__(
        self,
        resources: t.Collection[TorchResource],
        shape: t.Tuple[int],
        action: t.Union[AggregationAction, TorchAggregationFn],
        feature_store: FeatureStore,
    ):
        self._resources = resources
        self._shape = shape
        self._action = action
        self._feature_store = feature_store

    @property
    def execute(self) -> ResourceKey:
        """Sample of performing operations on resources... probably trash"""
        tensors: t.Tuple[torch.Tensor] = (r.value for r in self._resources)

        result: t.Optional[ResourceKey] = None
        if self._action == AggregationAction.VSTACK:
            result = torch.vstack(tensors)
        elif self._action == AggregationAction.VSTACK:
            result = torch.hstack(tensors)
        elif self._action == AggregationAction.SUM:
            result = torch.sum(tensors)
        elif isfunction(self._action):
            result = self._action(tensors)

        uuid = str(uuid.uuid4()).encode("utf-8")
        while self._feature_store[uuid] is not None:
            uuid = str(uuid.uuid4()).encode("utf-8")

        self._feature_store[uuid] = result
        return ResourceKey(uuid)


class TorchResourceTransform:

    def __init__(
        self,
        resource: TorchResource,
        shape: t.Tuple[int],
        action: t.Union[TransformAction, TorchTransformFn],
        feature_store: FeatureStore,
    ):
        self._resource = resource
        self._shape = shape
        self._action = action
        self._feature_store = feature_store

    @property
    def execute(self) -> ResourceKey:
        """Sample of performing operations on resources... probably trash"""
        # tensors: t.Tuple[torch.Tensor] = (key.value() for key in self._keys)
        tensor = self._key.value

        result: t.Optional[ResourceKey] = None
        if self._action == TransformAction.TRANSPOSE:
            result = torch.transpose(tensor)
        elif isfunction(self._action):
            result = self._action(tensor)

        uuid = str(uuid.uuid4()).encode("utf-8")
        while self._feature_store[uuid] is not None:
            uuid = str(uuid.uuid4()).encode("utf-8")

        self._feature_store[uuid] = result
        return ResourceKey(uuid)


# class ManagedMachineLearningPipeline(t.Protocol):
#     """Consider this name versus MLWorker?"""

#     def __init__(self): ...

#     def load(self, x):
#         return load.load(x)

#     def transform(self): ...

#     ...


class CoreMachineLearningWorker:
    """Basic functionality of ML worker that is shared across all worker types"""

    @staticmethod
    def fetch_model(model: ResourceKey) -> MachineLearningModelRef:
        """Given a ResourceKey, identify the physical location and model metadata"""
        # per Matt - just return bytes
        model.retrieve()

    @staticmethod
    def fetch_input(inputs: t.Collection[ResourceKey]) -> t.Collection[bytes]:
        """Given a collection of ResourceKeys, identify the physical location and input metadata"""
        data: t.List[bytes] = [input_.retrieve() for input_ in inputs]
        return data

    @staticmethod
    def batch_requests(
        data: t.Collection[Datum], batch_size: int
    ) -> t.Optional[t.Collection[Datum]]:
        """Create a batch of requests. Return the batch when batch_size datum have been
        collected or a configured batch duration has elapsed.

        Returns `None` if batch size has not been reached and timeout not exceeded."""
        ...

    @staticmethod
    def place_output(
        raw_keys: t.Collection[str], data: t.Collection[bytes], key_type: t.Type[ResourceKey]
        # need to know how to get back to original sub-batch inputs so they can be accurately placed, datum might need to include this.
    ) -> t.Optional[t.Collection[ResourceKey]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store"""
        for k, v in zip(raw_keys, data):
            key = key_type(k)
            key.put(v)


class MachineLearningWorker(t.Protocol):
    def deserialize(self, data_blob: bytes):
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend"""
        ...

    # def fetch_model(self, model: ResourceKey) -> MachineLearningModel:
    #     """Given a ResourceKey, identify the physical location and model metadata"""
    #     # per Matt - just return bytes
    #     ...

    def load_model(
        self, model: MachineLearningModelRef
    ) -> t.Any:  # MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into device memory"""
        # invoke separate API functions to put the model on GPU/accelerator (if exists)
        ...

    # def fetch_input(self, inputs: t.Collection[ResourceKey]) -> t.Collection[Datum]:
    #     """Given a collection of ResourceKeys, identify the physical location and input metadata"""
    #     ...

    def transform_input(
        self,
        data: t.Collection[Datum],
        # xform: TransformAction,
        # inplace: bool = False,
    ) -> t.Collection[Datum]:
        """Given a collection of data, perform a transformation on the data"""

    # def batch_requests(self, data: t.Collection[Datum], batch_size: int) -> t.Optional[t.Collection[Datum]]:
    #     """Create a batch of requests. Return the batch when batch_size datum have been
    #     collected or a configured batch duration has elapsed.

    #     Returns `None` if batch size has not been reached and timeout not exceeded."""

    def execute(
        self, model: MachineLearningModelRef, data: t.Collection[Datum]
    ) -> t.Any:
        """Execute an ML model on the given inputs"""

    def transform_output(
        self,
        data: t.Collection[
            Datum
        ],  # TODO: ask Al about assumption that "if i put in tensors, i will get out tensors. my generic probably fails here."
        xform: t.Union[
            TransformAction, t.Callable[[t.Collection[Datum]], t.Collection[Datum]]
        ],
        inplace: bool = False,
    ) -> t.Collection[Datum]:
        """Given a collection of data, perform a transformation on the data"""
        # TODO: determine if a single `transform_data` method can supply base feature (e.g.
        # pre-built transforms + option of sending transform callable)? No... this explicit
        # way gives the pipeline a definite pre and post call that are easy to understand

    # def place_output(
    #         self,
    #         data: t.Optional[t.Collection[Datum]],
    #         # need to know how to get back to original sub-batch inputs so they can be accurately placed, datum might need to include this.
    # ) -> t.Optional[t.Collection[ResourceKey]]:
    #     """Given a collection of data, make it available as a shared resource in the
    #     feature store"""

    # def serialize(self, data: t.Collection[Datum]) -> bytes:
    #     """Serialize the input data to bytes for transfer to client application"""

    @staticmethod
    def backend() -> str: ...


class MachineLearningWorkerBase(CoreMachineLearningWorker, MachineLearningWorker, ABC):
    @staticmethod
    @abstractmethod
    def deserialize(data_blob: bytes) -> t.Any:
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend"""
        ...

    @staticmethod
    @abstractmethod
    def load_model(model_ref: MachineLearningModelRef) -> t.Any:
        # model: MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into device memory"""
        # invoke separate API functions to put the model on GPU/accelerator (if exists)
        ...

    @staticmethod
    @abstractmethod
    def transform_input(
        data: t.Collection[Datum],
        # xform: TransformAction,
        # inplace: bool = False,
    ) -> t.Collection[Datum]:
        """Given a collection of data, perform a transformation on the data"""

    @staticmethod
    @abstractmethod
    def execute(model_ref: MachineLearningModelRef, data: t.Collection[Datum]) -> t.Any:
        """Execute an ML model on the given inputs"""

    @staticmethod
    @abstractmethod
    def transform_output(
        data: t.Collection[
            Datum
        ],  # TODO: ask Al about assumption that "if i put in tensors, i will get out tensors. my generic probably fails here."
        xform: t.Union[
            TransformAction, t.Callable[[t.Collection[Datum]], t.Collection[Datum]]
        ],
        inplace: bool = False,
    ) -> t.Collection[Datum]:
        """Given a collection of data, perform a transformation on the data"""
        # TODO: determine if a single `transform_data` method can supply base feature (e.g.
        # pre-built transforms + option of sending transform callable)? No... this explicit
        # way gives the pipeline a definite pre and post call that are easy to understand

    @staticmethod
    @abstractmethod
    def backend() -> str: ...


class DefaultTorchWorker(MachineLearningWorkerBase):
    @staticmethod
    def deserialize(data_blob: bytes) -> t.Any:
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend"""
        tensor: torch.Tensor = torch.load(io.BytesIO(data_blob))
        return tensor

    @staticmethod
    def load_model(model_ref: MachineLearningModelRef) -> torch.nn.Module:
        # MLMLocator? something that doesn't say "I am actually the model"
        """Given a loaded MachineLearningModel, ensure it is loaded into device memory"""
        # invoke separate API functions to put the model on GPU/accelerator (if exists)
        raw_bytes = model_ref.model()
        model_bytes = io.BytesIO(raw_bytes)
        model = torch.load(model_bytes)
        return model

    @staticmethod
    def transform_input(
        data: t.Collection[Datum],
    ) -> t.Collection[Datum]:
        """Given a collection of data, perform a no-op transformation on the data"""
        return [torch.Tensor.copy_(item, False) for item in data]
        # return data # note: this fails copy test!

    @staticmethod
    def execute(model_ref: MachineLearningModelRef, data: t.Collection[Datum]) -> t.Any:
        """Execute an ML model on the given inputs"""
        model = DefaultTorchWorker.load_model(model_ref)
        results = [model(tensor) for tensor in data]
        return results

    @staticmethod
    def transform_output(
        data: t.Collection[Datum],
        # TODO: ask Al about assumption that "if i put in tensors, i will get out tensors. my generic probably fails here."
        # xform: t.Union[
        #     TransformAction, t.Callable[[t.Collection[Datum]], t.Collection[Datum]]
        # ],
        # inplace: bool = False,
    ) -> t.Collection[Datum]:
        """Given a collection of data, perform a transformation on the data"""
        # TODO: determine if a single `transform_data` method can supply base feature (e.g.
        # pre-built transforms + option of sending transform callable)? No... this explicit
        # way gives the pipeline a definite pre and post call that are easy to understand
        return [torch.Tensor.copy_(item, False) for item in data]

    @staticmethod
    def backend() -> str:
        return "PyTorch"


# # class TorchWorker(CoreMachineLearningWorker, MachineLearningWorker):
# class TorchWorker(MachineLearningWorker):
#     def __init__(self, work_queue: mp.Queue) -> None:
#         self._work_queue = work_queue

#     @property
#     def work_queue(self) -> mp.Queue:
#         # i think i'm just going to call this crap directly instead of queueing to the worker
#         return self._work_queue

#     @staticmethod
#     def backend() -> str:
#         return "PyTorch"


# # class TensorflowWorker(MachineLearningWorker): ...


# class ServiceHost(ABC):
#     """Nice place to have some default entrypoint junk (args, event loop, cooldown, etc)"""

#     def __init__(self, as_service: bool = False, cooldown: int = 0) -> None:
#         self._as_service = as_service
#         """If the service should run until shutdown function returns True"""
#         self._cooldown = cooldown
#         """Duration of a cooldown period between requests to the service before shutdown"""

#     @abstractmethod
#     def _on_execute_iteration(self, timestamp: int) -> None: ...

#     @abstractmethod
#     def _can_shutdown(self) -> bool: ...

#     def _on_start(self) -> None:
#         print(f"Starting {self.__class__.__name__}")

#     def _on_shutdown(self) -> None:
#         print(f"Shutting down {self.__class__.__name__}")

#     def _on_cooldown(self) -> None:
#         print(f"Cooldown exceeded by {self.__class__.__name__}")

#     def execute(self, work_queue: mp.Queue) -> None:
#         self._on_start()

#         ts = time.time_ns()
#         last_ts = ts
#         running = True
#         elapsed_cooldown = 0
#         NS_TO_S = 1000000000
#         cooldown_ns = self._cooldown * NS_TO_S

#         # if we're run-once, use cooldown to short circuit
#         if not self._as_service:
#             self._cooldown = 1
#             last_ts = ts - (cooldown_ns * 2)

#         while running:
#             self._on_execute_iteration(ts)

#             eligible_to_quit = self._can_shutdown()

#             if self._cooldown and not eligible_to_quit:
#                 # reset timer any time cooldown is interrupted
#                 elapsed_cooldown = 0

#             # allow service to shutdown if no cooldown period applies...
#             running = not eligible_to_quit

#             # ... but verify we don't have remaining cooldown time
#             if self._cooldown:
#                 elapsed_cooldown += ts - last_ts
#                 remaining = cooldown_ns - elapsed_cooldown
#                 running = remaining > 0

#                 rem_in_s = remaining / NS_TO_S

#                 if not running:
#                     cd_in_s = cooldown_ns / NS_TO_S
#                     print(f"cooldown {cd_in_s}s exceeded by {abs(rem_in_s):.2f}s")
#                     self._on_cooldown()
#                     continue
#                 else:
#                     print(f"cooldown remaining {abs(rem_in_s):.2f}s")

#             last_ts = ts
#             ts = time.time_ns()
#             time.sleep(1)

#         self._on_shutdown()


# _MachineLearningModel = t.TypeVar("_MachineLearningModel")


# class ModelContainer(t.Generic[_MachineLearningModel]):
#     """wrapper around concrete models"""

#     def __init__(self, model: _MachineLearningModel) -> None:
#         self._model: _MachineLearningModel = model


# class ModelLoader(ABC):

#     @singledispatchmethod
#     @classmethod
#     @abstractmethod
#     def load(self, model: bytes) -> ModelContainer:
#         print(f"loading model from bytes {bytes[:10]}...")
#         raise NotImplementedError("Loading a model from bytes is not supported")

#     @load.register
#     @classmethod
#     def _load_path(self, path: pathlib.Path) -> ModelContainer:
#         print(f"loading model from {path}")
#         model_bytes = path.read_bytes()
#         return ModelLoader.load(model_bytes)

#     @load.register
#     @classmethod
#     def _load_key(self, key: ResourceKey) -> ModelContainer:
#         print(f"loading model with key: {key}")
#         # raise NotImplementedError("Loading a model by ResourceKey is not supported")
#         model_bytes = key.retrieve()
#         return ModelLoader.load(model_bytes)


# class TorchModelLoader(ABC):

#     @singledispatchmethod
#     @classmethod
#     @abstractmethod
#     def load(self, model: bytes) -> ModelContainer:
#         print(f"loading model from bytes {bytes[:10]}...")
#         raise NotImplementedError("Loading a model from bytes is not supported")


# class InferenceRequest:

#     def __init__(
#         self,
#         backend: t.Optional[str] = None,
#         model: t.Optional[MachineLearningModel] = None,
#         callback: t.Optional[CommChannel] = None,
#         value: t.Optional[bytes] = None,
#     ):
#         self.backend = backend
#         self.model = model
#         self.callback = callback
#         self.value = value

#     @classmethod
#     def from_msg(cls, msg: bytes) -> "t.Optional[InferenceRequest]":
#         msg_str = msg.decode("utf-8")
#         # todo: we'll need to filter properly ... or assume all messages are inference reqs
#         if not ":" in msg_str or not msg_str.startswith("PyTorch"):
#             return None

#         prefix, model_path, serialized_input, serialized_channel = msg_str.split(
#             ":", maxsplit=3
#         )

#         # callback = CommChannel.find(serialized_channel)
#         # key = FileSystemKey(f"{prefix}:{model_name}".encode("utf-8"))

#         key = FileSystemKey(pathlib.Path(model_path))

#         persistence_path = pathlib.Path(serialized_channel)
#         callback = FileCommChannel(persistence_path)

#         model = MachineLearningModel(prefix, key)
#         return InferenceRequest(prefix, model, callback, serialized_input)


# class WorkerManager(ServiceHost):

#     def __init__(
#         self, feature_store: FeatureStore, as_service: bool = False, cooldown: int = 0
#     ) -> None:
#         super().__init__(as_service, cooldown)

#         self._workers: t.Dict[str, MachineLearningWorker] = {}
#         """a collection of workers the manager is controlling"""
#         self._upstream_queue: t.Optional[mp.Queue] = None
#         """the queue the manager monitors for new tasks"""
#         self._feature_store: FeatureStore = feature_store
#         """a feature store to retrieve models from"""

#     def _on_execute_iteration(self, timestamp: int) -> None:
#         print(f"{timestamp} executing worker manager pipeline")

#         if self.upstream_queue is None:
#             print("No queue to check for tasks")
#             return

#         msg: str = self.upstream_queue.get()

#         if request := InferenceRequest.from_msg(msg):
#             worker = self._workers.get(request.model.backend, None)
#             if request.model.backend == "PyTorch":
#                 print(f"Request received for {request.model.backend} backend")
#                 model = self._feature_store[request.model._key.key]

#                 if not model:
#                     print("No model found in feature_store")
#                     # START hack! this really needs to come from message but for now, i'll use demo model
#                     model_bytes = request.model._key.retrieve()
#                     # with pathlib.Path("./demo-model.pt") as model_file:
#                     #     resource_key.put(model_file)
#                     # END hack!

#                     model = MachineLearningModel(
#                         request.model.backend,
#                         request.model._key,
#                     )
#                     self._feature_store[request.model._key.key] = model

#                     # note: if the req is direct inference, request.model could be
#                     # populated and need to be put _INTO_ the feature store...

#             if not worker:
#                 print("Adding new TorchWorker to managed workers")
#                 downstream_queue = mp.Queue()
#                 worker = TorchWorker(model, downstream_queue)
#                 self.add_worker(worker, downstream_queue)

#         # perform the inference pipeline with a worker
#         worker, _ = self._workers.get(request.model.backend, None)
#         if worker:
#             print(f"Retrieved {request.model.backend} worker from managed workers")
#             value: torch.Tensor = worker.infer(request.value)
#             value_as_bytes = str(value).encode("utf-8")

#             # self._deserialize_channel_descriptor(request.callback)
#             callback_channel = request.callback
#             callback_channel.send(value_as_bytes)
#         else:
#             print(f"Cannot retrieve {request.model.backend} worker")

#     def _can_shutdown(self) -> bool:
#         return bool(self._workers)

#     @property
#     def upstream_queue(self) -> t.Optional[mp.Queue]:
#         return self._upstream_queue

#     @upstream_queue.setter
#     def upstream_queue(self, value: mp.Queue) -> None:
#         self._upstream_queue = value

#     def add_worker(self, worker: MachineLearningWorker, work_queue: mp.Queue) -> None:
#         self._workers[worker.model.backend] = (worker, work_queue)

#     def _deserialize_channel_descriptor(self, value: bytes) -> CommChannel:
#         channel = FileCommChannel.find(
#             value
#         )  # todo: inject CommChannels based on messages...
#         # channel.send(value)
#         return channel


# class MachineLearningModelWorker(ServiceHost):
#     def __init__(
#         self, model: MachineLearningModel, work_queue: mp.Queue, backend: str
#     ) -> None:
#         self._model: MachineLearningModel = model
#         self._work_queue: mp.Queue = work_queue
#         self._supported_backend: str = backend

#         if self.work_queue is None:
#             raise ValueError("No work queue to check for tasks")
#         # self._callback: t.Any = None

#     @property
#     def model(self) -> MachineLearningModel:
#         return self._model

#     @property
#     def work_queue(self) -> mp.Queue:
#         return self._work_queue

#     @property
#     def supported_backend(self) -> str:
#         return self._supported_backend

#     @abstractmethod
#     def _hydrate_model(self, raw_model: bytes) -> t.Any:
#         """Take the bytes of the model and convert to a model for the given backend"""
#         ...

#     def publish_result(self, result: torch.Tensor) -> None:
#         callback = None

#         buffer = io.BytesIO()
#         torch.save(result, buffer)
#         # callback.send_bytes(buffer.read())

#     @abstractmethod
#     def infer(self, input: bytes) -> t.Any: ...

#     def _can_shutdown(self) -> bool:
#         return True


# class TorchWorker(MachineLearningModelWorker):
#     def __init__(self, model: MachineLearningModel, work_queue: mp.Queue) -> None:
#         super().__init__(model, work_queue, "PyTorch")
#         self._torch_model: t.Optional[torch.Module] = None

#     def _on_execute_iteration(self, timestamp: int) -> None:
#         print(f"{timestamp} executing worker manager pipeline")

#         msg: str = self.work_queue.get()
#         print(f"{self.__class__.__name} received msg: {msg}")

#         self.infer(torch.Tensor([1, 2]))

#     def _on_start(self) -> None:
#         print(f"Starting {self.__class__.__name__}")
#         if self._torch_model is None:
#             model_bytes = self._model._key.retrieve()
#             self._torch_model: torch.Module = self._hydrate_model(model_bytes)

#     def _hydrate_model(self, raw_model: bytes) -> t.Any:
#         """Take the bytes of the model and convert to a model for the given backend"""
#         # todo: real hydrate from raw...
#         model = torch.nn.Sequential(torch.nn.Linear(2, 1))
#         return model

#     def infer(self, input: bytes) -> torch.Tensor:
#         model_bytes = self._model._key.retrieve()
#         torch_model: torch.Module = self._hydrate_model(model_bytes)
#         input = torch.Tensor([1, 2])

#         # result = torch_model(input)
#         result = torch.Tensor(999)
#         self.publish_result(result)
#         return result


# def mock_work(worker_manager_queue: mp.Queue) -> None:
#     while True:
#         time.sleep(1)
#         # 1. for demo, ignore upstream and just put stuff into downstream
#         # 2. for demo, only one downstream but we'd normally have to filter
#         #       msg content and send to the correct downstream (worker) queue
#         ts = time.time_ns()
#         test_dir = "/lus/bnchlu1/mcbridch/code/ss/tests/test_output/brainstorm"
#         test_path = pathlib.Path(test_dir)

#         mock_channel = test_path / f"brainstorm-{ts}.txt"
#         mock_model = test_path / "brainstorm.pt"

#         test_path.mkdir(parents=True, exist_ok=True)
#         mock_channel.touch()
#         mock_model.touch()

#         msg = f"PyTorch:{mock_model}:MockInputToReplace:{mock_channel}"
#         worker_manager_queue.put(msg.encode("utf-8"))


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

    # upstream_queue = mp.Queue()  # the incoming messages from application
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
    # msg_pump = mp.Process(target=mock_work, args=(upstream_queue,))
    # msg_pump.start()

    # # create a process to process commands
    # process = mp.Process(target=worker_manager.execute, args=(time.time_ns(),))
    # process.start()
    # process.join()

    # msg_pump.kill()
    backend = DefaultTorchWorker.backend()
    print(f"{backend=}")
