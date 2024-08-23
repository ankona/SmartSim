import os
import subprocess as sp
import threading as mp
import typing as t

import smartsim._core.mli.infrastructure.storage.backbonefeaturestore as bb
from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import dragon_ddict

# import dragon


# isort: off
# pylint: disable-next=import-error
# from dragon.channels import Channel

# isort: on


#########
# BACKEND
#########


def register_consumer(
    backbone: bb.BackboneFeatureStore,
    name: t.Optional[str] = None,
    filters: t.Optional[t.List[bb.EventCategory]] = None,
) -> bb.EventConsumer:
    # create my event notification channel
    consumer_channel = DragonCommChannel.from_local()
    consumer = bb.EventConsumer(consumer_channel, backbone, filters, name=name)

    registration_iter = consumer.register()
    is_registered = False

    while not is_registered:
        is_registered = next(registration_iter)

    return consumer


def attach_to_backbone() -> bb.BackboneFeatureStore:
    # connect to the backbone based on the env var (or CLI arg)
    # holding backbone descriptor
    # NOTE: this will be wrong format (str vs bytes, b64). probably
    # need to create `class Descriptor`
    backbone_descriptor: t.Optional[str] = os.environ.get(
        "_SMARTSIM_INFRA_BACKBONE", None
    )
    if not backbone_descriptor:
        raise ValueError("No backbone descriptor available")

    storage = dragon_ddict.DDict.attach(backbone_descriptor)
    return bb.BackboneFeatureStore(storage)


def on_consumer_registered_callaback(backbone_descriptor: str) -> None:
    """Perform an infinite loop that waits to receive registration messages
    and publishes the new consumer information to the backbone feature store"""

    # attach to the backbone
    backbone = bb.BackboneFeatureStore.from_writable_descriptor(backbone_descriptor)

    # create the backend registrar callback channel...
    back_channel = DragonCommChannel.from_local()
    consumer = bb.EventConsumer(
        back_channel, backbone, [bb.EventCategory.CONSUMER_CREATED]
    )

    # ... and push the callback channel into the backbone
    backbone.backend_channel = back_channel.descriptor_string

    while True:
        # block until new event(s) received
        # registration_events = back_channel.recv()
        registrations: t.List[bb.EventBase] = consumer.receive()

        # de-duplicate registration messages
        descriptors = set(
            str(getattr(reg, "descriptor", "unk")) for reg in registrations
        )
        action_map = {
            str(getattr(reg, "descriptor", "unk")): str(getattr(reg, "filters", []))
            for reg in registrations
            if str(getattr(reg, "descriptor", "unk")) in descriptors
        }

        # publish the consumer list to the backbone
        backbone.notification_channels = tuple(action_map.keys())

        # publish the subscription details to backbone
        for descriptor, filters in action_map.items():
            backbone[descriptor] = filters


# assume backbone descriptor is passed to backend via env
the_backbone = attach_to_backbone()
consumer_thread = mp.Thread(
    target=on_consumer_registered_callaback, args=(the_backbone.descriptor,)
)
consumer_thread.start()

########
# WMGR 1
########

# the worker manager


def load_model(_key: str) -> None:
    ...


def on_model_update(event: bb.OnWriteFeatureStore) -> None:
    my_key = "my-model-key"

    if event.key == my_key:
        load_model(event.key)


the_backbone = attach_to_backbone()
the_consumer = register_consumer(
    the_backbone, "worker-manager-XXX", [bb.EventCategory.FEATURE_STORE_WRITTEN]
)
consumer_thread = mp.Thread(target=the_consumer.listen, args=(on_model_update,))
consumer_thread.start()


########
# APP
########

# the app is responsible for creating a client that
# bootstraps all of the SmartSim/MLI resources


class Client:
    def __init__(self) -> None:
        self.backbone: t.Optional[bb.BackboneFeatureStore] = None
        self.publisher: t.Optional[bb.EventBroadcaster] = None
        self.backend: t.Optional[sp.Popen[bytes]] = None

    def create_backbone(self) -> bb.BackboneFeatureStore:
        # create the backbone
        storage = dragon_ddict.DDict()
        self.backbone = bb.BackboneFeatureStore(storage, allow_write=False)
        return self.backbone

    def create_publisher(self) -> bb.EventBroadcaster:
        if not self.backbone:
            raise ValueError("Backbone was not set. Unable to create publisher")

        broadcaster = bb.EventBroadcaster(
            self.backbone, DragonCommChannel.from_descriptor
        )
        return broadcaster

    def _allocate_backend(self, backbone: bb.BackboneFeatureStore) -> None:
        """Do all the backend stuff above... Client only knows the backbone"""
        # trigger backend script
        if not self.backbone:
            raise ValueError("Backbone was not set. Unable to create publisher")

        # pylint: disable-next=consider-using-with
        self.backend = sp.Popen(
            ["python", "/path/to/dragon/backend.py"],
            env={"_SMARTSIM_INFRA_BACKBONE": backbone.descriptor},
        )

    def init(self) -> None:
        self.backbone = self.create_backbone()
        self.publisher = self.create_publisher()
        self._allocate_backend(self.backbone)

    def set_model(self, key: str, model: bytes) -> None:
        # pretend we wrote the model into some feature store key
        feature_store = {}
        feature_store[key] = model

        if not self.publisher:
            raise ValueError("Publisher was not set. Unable to send")

        self.publisher.send(bb.OnWriteFeatureStore("mock-fs-descriptor", key))


client = Client()
client.init()
