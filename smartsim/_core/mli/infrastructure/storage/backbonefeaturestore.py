# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import abc, defaultdict, deque
import enum
import typing as t

from attr import dataclass

# pylint: disable=import-error
# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.featurestore import ReservedKeys
from smartsim.log import get_logger

logger = get_logger(__name__)


# todo: did i create an arms race where a developer just grabs the backbone
# and passes it wherever they need a FeatureStore?
class BackboneFeatureStore(DragonFeatureStore):
    """A DragonFeatureStore wrapper with utility methods for accessing shared
    information stored in the MLI backbone feature store"""

    def __init__(self, storage: "dragon_ddict.DDict") -> None:
        """Initialize the DragonFeatureStore instance

        :param storage: A distributed dictionary to be used as the underlying
        storage mechanism of the feature store"""
        super().__init__(storage)
        self._reserved_write_enabled = True

    @property
    def notification_channels(self) -> t.Sequence[str]:
        """Retrieve descriptors for all registered MLI notification channels

        :returns: the list of descriptors"""
        if ReservedKeys.MLI_NOTIFY_CONSUMERS in self:
            return self[ReservedKeys.MLI_NOTIFY_CONSUMERS]
        return []

    @notification_channels.setter
    def notification_channels(
        self, value: t.Sequence[str]
    ) -> t.Sequence[CommChannelBase]:
        self[ReservedKeys.MLI_NOTIFY_CONSUMERS] = value


class EventTypes(str, enum.Enum):
    """Predefined event types raised by SmartSim backend"""

    CONSUMER_CREATED: str = enum.auto()
    FEATURE_STORE_WRITTEN: str = enum.auto()
    UNKNOWN: str = enum.auto()


@dataclass
class EventBase:
    type: EventTypes
    """The event category for this event; may be used for addressing,
    prioritization, or filtering of events by a event publisher/consumer"""

    def __str__(self) -> str:
        """Convert the event to a string"""
        return f"{type}"

    def __bytes__(self) -> bytes:
        """Default conversion to bytes for an event required to publish
        messages using byte-oriented communication channels"""
        # todo: consider just using pickle
        return bytes(str(self), encoding="utf-8")


class OnCreateConsumer(EventBase):
    """Publish this event when a new event consumer registration is required"""

    descriptor: str
    """The descriptor of the comm channel exposed by the event consumer"""

    def __init__(self, descriptor: str) -> None:
        """Initialize the event instance

        :param type: the event category"""
        super().__init__(EventTypes.CONSUMER_CREATED)
        self.descriptor = descriptor

    def __str__(self) -> str:
        """Convert the event to a string"""
        # todo: consider just using pickle
        return f"{str(super())}|{self.descriptor}"


class OnWriteFeatureStore(EventBase):
    """Publish this event when a feature store key is written"""

    descriptor: str
    """The descriptor of the feature store where the write occurred"""
    key: str
    """The key for where the write occurred"""

    def __init__(self, descriptor: str, key: str) -> None:
        super().__init__(EventTypes.FEATURE_STORE_WRITTEN)
        self.descriptor = descriptor
        self.key = key

    def __str__(self) -> str:
        """Convert the event to a string"""
        # todo: consider just using pickle
        return f"{str(super())}|{self.descriptor}|{self.key}"


class EventPublisher(t.Protocol):
    """Core API of a class that publishes events"""

    def send(self, event: EventBase) -> int:
        """The send operation"""


class EventBroadcaster:
    """Performs fan-out publishing of system events"""

    def __init__(
        self,
        backbone: BackboneFeatureStore,
        channel_factory: t.Optional[t.Callable[[str], CommChannelBase]] = None,
        buffer_size: int = 100,
    ) -> None:
        """Initialize the EventPublisher instance

        :param backbone:
        :param channel_factory:
        :param buffer_size: Maximum number of events to store before discarding.
        Discards oldest events first.
        """
        self._backbone = backbone
        """The backbone feature store used to retrieve consumer descriptors"""
        self._channel_factory = channel_factory
        """A factory method used to instantiate channels from descriptors"""
        self._channel_cache: t.Dict[str, t.Optional[CommChannelBase]] = defaultdict(
            lambda: None
        )
        """A mapping of instantiated channels that can be re-used. Automatically 
        calls the channel factory if a descriptor is not already in the collection"""
        self._event_buffer: t.Deque[EventBase] = deque(maxlen=buffer_size)
        """A buffer for storing events when a consumer list is not found. Buffer size
        can be fixed by passing the `buffer_size` parameter."""
        self._num_discards: int = 0
        """Number of messages discarded from the buffer since the last successful
        broadcast"""
        self._buffer_size: int = buffer_size
        """Maximum size of the event buffer"""
        self._descriptors: t.Set[str]
        """Stores the most recent list of broadcast consumers. Updated automatically
        on each broadcast"""

    @property
    def num_buffered(self) -> int:
        return len(self._event_buffer)

    @property
    def num_discarded(self) -> int:
        return self._num_discards

    def _save_to_buffer(self, event: EventBase) -> None:
        """Places a new event in the buffer until full. Continues adding events to
        the buffer using a first-in, first-discarded strategy."""
        if len(self._event_buffer) == self._buffer_size:
            self._event_buffer.popleft()
            self._num_discards += 1

        self._event_buffer.append(event)

    def _log_broadcast_start(self) -> None:
        """Logs broadcast statistics"""
        num_pending = len(self._event_buffer)
        num_consumers = len(self._descriptors)
        logger.debug(f"Broadcasting {num_pending} events to {num_consumers} consumers")

    def _prune_unused_consumers(self) -> None:
        """Performs maintenance on the channel cache by pruning any channel
        that has been removed from the consumers list"""
        active_consumers = set(self._descriptors)
        current_channels = set(self._channel_cache.keys())

        # find any cached channels that are now unused
        inactive_channels = current_channels.difference(active_consumers)
        new_channels = active_consumers.difference(current_channels)

        for descriptor in inactive_channels:
            self._channel_cache.pop(descriptor)

        logger.debug(
            f"Pruning {len(inactive_channels)} stale consumer channels"
            f" and found {len(new_channels)} new channels"
        )

    def _broadcast(self, event: EventBase) -> int:
        """Broadcasts an event to all registered event consumers.

        :param event: an event to publish
        :return: the number of events broadcasted to consumers"""
        self._save_to_buffer(event)

        self._descriptors = set(self._backbone.notification_channels)
        if not self._descriptors:
            logger.warning("No event consumers are registered")
            return 0

        if not self._channel_factory:
            logger.warning("No channel factory provided for consumers")
            return 0

        self._prune_unused_consumers()
        self._log_broadcast_start()

        num_sent: int = 0

        for descriptor in map(str, self._descriptors):
            try:
                while event := self._event_buffer.pop():
                    event_bytes = bytes(event)

                    comm_channel = self._channel_cache[descriptor]
                    if comm_channel is None:
                        comm_channel = self._channel_factory(descriptor)
                        self._channel_cache[descriptor] = comm_channel
                    comm_channel.send(event_bytes)
                    num_sent += 1
                    self._num_discards = 0
            except IndexError:
                logger.debug(f"Event buffer exhausted")
            except KeyError:
                logger.error(
                    f"Cannot broadcast to unknown channel: {descriptor}", exc_info=True
                )
            except Exception as ex:
                logger.error(
                    f"Broadcast failed for channel: {descriptor}", exc_info=True
                )

        return num_sent

    def send(self, event: EventBase) -> int:
        """Implementation of `send` method of the `EventPublisher` protocol. Publishes
        the supplied event to all registered broadcast consumers

        :param event: an event to publish
        :returns: the number of events successfully published"""
        return self._broadcast(event)


class EventConsumer:
    """Reads system events published to a communications channel"""

    def __init__(
        self,
        comm_channel: CommChannelBase,
        backbone: BackboneFeatureStore,
        filters: t.Optional[t.List[EventTypes]],
    ) -> None:
        """Initialize the EventConsumer instance

        :param comm_channel: communications channel to listen to for events
        :param backbone: the MLI backbone feature store
        :param filters: a list of event types to deliver. when empty, all
        events will be delivered"""
        self._comm_channel = comm_channel
        self._backbone = backbone
        self._global_filters = filters or []

    def receive(
        self, filters: t.Optional[t.List[EventTypes]] = None
    ) -> t.List[EventBase]:
        """Receives available published event(s)

        :param filters: additional filters to add to the global filters configured
        on the EventConsumer instance
        :returns: a list of events that pass any configured filters"""
        if filters is None:
            filters = []

        filter_set = [*self._global_filters, *filters]
        messages: t.List[t.Any] = []

        # todo: need to be able to receive a bunch of messages at once, not just one
        while msg:
            msg: EventBase = self._comm_channel.recv()  # todo: need a timeout here
            if not msg or len(msg) == 0:
                break

            # skip any messages not passing a filter
            if filter_set and msg.type not in filter_set:
                continue

            messages.append(msg)

        return messages
