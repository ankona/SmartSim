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
import os
import os.path as osp
import pathlib
import shutil
import typing as t

import pytest

from smartsim import Experiment
from smartsim._core.config import CONFIG
from smartsim._core.config.config import Config
from smartsim._core.mli.infrastructure.storage.backbonefeaturestore import (
    BackboneFeatureStore,
    EventBroadcaster,
    OnCreateConsumer,
)
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.featurestore import ReservedKeys
from smartsim._core.utils import serialize
from smartsim.database import Orchestrator
from smartsim.entity import Model
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import RunSettings
from smartsim.status import SmartSimStatus
from tests.mli.channel import FileSystemCommChannel
from tests.mli.featurestore import FileSystemFeatureStore, MemoryFeatureStore

if t.TYPE_CHECKING:
    import conftest

dragon = pytest.importorskip("dragon")


# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


def test_mli_reserved_keys_writes() -> None:
    """Verify that attempts to write to reserved keys are blocked from a
    standard DragonFeatureStore but enabled with the BackboneFeatureStore"""

    mock_storage = {}
    dfs = DragonFeatureStore(mock_storage)
    backbone = BackboneFeatureStore(mock_storage)
    other = MemoryFeatureStore(mock_storage)

    expected_value = "value"

    for reserved_key in ReservedKeys:
        # we expect every reserved key to fail using DragonFeatureStore...
        with pytest.raises(SmartSimError) as ex:
            dfs[reserved_key] = expected_value

        assert "reserved key" in ex.value.args[0]

        # ... and expect other feature stores to respect reserved keys
        with pytest.raises(SmartSimError) as ex:
            other[reserved_key] = expected_value

        assert "reserved key" in ex.value.args[0]

        # ...and those same keys to succeed on the backbone
        backbone[reserved_key] = expected_value
        actual_value = backbone[reserved_key]
        assert actual_value == expected_value


def test_mli_consumers_read_by_key() -> None:
    """Verify that the value returned from the mli consumers
    method is written to the correct key and reads are
    allowed via standard dragon feature store.
    NOTE: should reserved reads also be blocked"""

    mock_storage = {}
    dfs = DragonFeatureStore(mock_storage)
    backbone = BackboneFeatureStore(mock_storage)
    other = MemoryFeatureStore(mock_storage)

    expected_value = "value"

    # write using backbone that has permission to write reserved keys
    backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS] = expected_value

    # confirm read-only access to reserved keys from any FeatureStore
    for fs in [dfs, backbone, other]:
        assert fs[ReservedKeys.MLI_NOTIFY_CONSUMERS] == expected_value


def test_mli_consumers_read_by_backbone() -> None:
    """Verify that the backbone reads the correct location
    when using the backbone feature store API instead of mapping API"""

    mock_storage = {}
    backbone = BackboneFeatureStore(mock_storage)
    expected_value = "value"

    backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS] = expected_value

    # confirm reading via convenience method returns expected value
    assert backbone.notification_channels[0] == expected_value


def test_mli_consumers_write_by_backbone() -> None:
    """Verify that the backbone writes the correct location
    when using the backbone feature store API instead of mapping API"""

    mock_storage = {}
    backbone = BackboneFeatureStore(mock_storage)
    expected_value = ["value"]

    backbone.notification_channels = expected_value

    # confirm write using convenience method targets expected key
    assert backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS] == ",".join(expected_value)


def test_eventpublisher_broadcast_no_factory(test_dir: str) -> None:
    """Verify that a broadcast operation without any registered subscribers
    succeeds without raising Exceptions

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}
    consumer_descriptor = storage_path / "test-consumer"

    # NOTE: we're not putting any consumers into the backbone here!
    backbone = BackboneFeatureStore(mock_storage)

    event = OnCreateConsumer(consumer_descriptor)
    buffer_size = 20

    publisher = EventBroadcaster(backbone, buffer_size=buffer_size)
    num_receivers = 0

    # publishing this event without any known consumers registered should succeed
    # but report that it didn't have anybody to send the event to
    consumer_descriptor = storage_path / f"test-consumer"
    event = OnCreateConsumer(consumer_descriptor)

    num_receivers += publisher.send(event)

    # confirm no changes to the backbone occur when fetching the empty consumer key
    key_in_features_store = ReservedKeys.MLI_NOTIFY_CONSUMERS in backbone
    assert not key_in_features_store

    # confirm that the broadcast reports no events published
    assert num_receivers == 0
    # confirm that the broadcast buffered the event for a later send
    assert publisher.num_buffered == 1
    # confirm that no events were discarded from the buffer
    assert publisher.num_discarded == 0


def test_eventpublisher_broadcast_buffer_not_exceeded(test_dir: str) -> None:
    """Verify that a broadcast continues to grow the buffer size until
    the max buffer size is reached

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}
    consumer_descriptor = storage_path / "test-consumer"

    # NOTE: we're not putting any consumers into the backbone here!
    backbone = BackboneFeatureStore(mock_storage)

    event = OnCreateConsumer(consumer_descriptor)
    buffer_size = 20

    publisher = EventBroadcaster(backbone, buffer_size=buffer_size)
    num_receivers = 0

    # publishing this event without any known consumers registered should succeed
    # but report that it didn't have anybody to send the event to
    num_to_send = buffer_size
    for i in range(num_to_send):
        consumer_descriptor = storage_path / f"test-consumer-{i}"
        event = OnCreateConsumer(consumer_descriptor)

        num_receivers += publisher.send(event)

    # confirm no changes to the backbone occur when fetching the empty consumer key
    key_in_features_store = ReservedKeys.MLI_NOTIFY_CONSUMERS in backbone
    assert not key_in_features_store

    # confirm that the broadcast reports no events published
    assert num_receivers == 0
    # confirm that the broadcast buffered the event for a later send and
    # wasn't constrained by the buffer size
    assert publisher.num_buffered == num_to_send
    # confirm that no events were discarded from the buffer
    assert publisher.num_discarded == 0


@pytest.mark.parametrize(
    "buffer_size,num_to_send",
    [
        pytest.param(20, 20, id="full buffer, no discards"),
        pytest.param(20, 21, id="full buffer, 1 discard"),
        pytest.param(20, 24, id="full buffer, multiple discards"),
        pytest.param(20, 40, id="discard entire buffer 1x"),
        pytest.param(20, 70, id="discard buffer 3x"),
    ],
)
def test_eventpublisher_broadcast_buffer_discard(
    test_dir: str,
    buffer_size: int,
    num_to_send: int,
) -> None:
    """Verify that a broadcast operation discards events once the configured
    buffer size is reached, then discards the oldest messages first

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param buffer_size: how large the buffer should be in this run
    :param num_to_send: how many events to broadcast
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}
    consumer_descriptor = storage_path / "test-consumer"

    # NOTE: we're not putting any consumers into the backbone here!
    # this should causes rolling buffering we can verify
    backbone = BackboneFeatureStore(mock_storage)
    event = OnCreateConsumer(consumer_descriptor)
    publisher = EventBroadcaster(backbone, buffer_size=buffer_size)
    num_receivers = 0

    all_events = []

    # raise more events than the buffer can hold
    for i in range(num_to_send):
        consumer_descriptor = storage_path / str(i)
        event = OnCreateConsumer(consumer_descriptor)
        all_events.append(event)

        num_receivers += publisher.send(event)

    # confirm that the buffer does not grow beyond configured maximum
    assert publisher.num_buffered == buffer_size

    # confirm that the number discarded grows unbounded
    assert publisher.num_discarded == num_to_send - buffer_size

    # confirm that the buffer is populated with the last `buffer_size` events
    expected_start_idx = num_to_send - buffer_size
    for i in range(buffer_size):
        actual_idx = expected_start_idx + i

        buffer_value = publisher._event_buffer[i].descriptor
        overall_value = all_events[actual_idx].descriptor

        assert buffer_value == overall_value


def test_eventpublisher_broadcast_to_empty_consumer_list(test_dir: str) -> None:
    """Verify that a broadcast operation without any registered subscribers
    succeeds without raising Exceptions

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    # prep our backbone with a consumer list
    backbone = BackboneFeatureStore(mock_storage)
    backbone.notification_channels = (consumer_descriptor,)

    event = OnCreateConsumer(consumer_descriptor)
    publisher = EventBroadcaster(backbone)
    num_receivers = publisher.send(event)

    registered_consumers = backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS]

    # confirm that no consumers exist in backbone to send to
    assert registered_consumers
    # confirm that the broadcast reports no events published
    assert num_receivers == 0
    # confirm that the broadcast buffered the event for a later send
    assert publisher.num_buffered == 1
    # confirm that no events were discarded from the buffer
    assert publisher.num_discarded == 0


def test_eventpublisher_broadcast_empties_buffer(test_dir: str) -> None:
    """Verify that a successful broadcast clears messages from the event
    buffer when a new message is sent and consumers are registered

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    backbone = BackboneFeatureStore(mock_storage)
    backbone.notification_channels = (consumer_descriptor,)

    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    # mock building up some buffered events
    num_buffered_events = 14
    for i in range(num_buffered_events):
        event = OnCreateConsumer(storage_path / f"test-consumer-{str(i)}")
        publisher._event_buffer.append(event)

    event0 = OnCreateConsumer(
        storage_path / f"test-consumer-{str(num_buffered_events + 1)}"
    )

    num_receivers = publisher.send(event0)
    # 1 receiver x 15 total events == 15 events
    assert num_receivers == num_buffered_events + 1
    assert publisher.num_discarded == 0


@pytest.mark.parametrize(
    "num_consumers, num_buffered, expected_num_sent",
    [
        pytest.param(0, 7, 0, id="0 x (7+1) - no consumers, multi-buffer"),
        pytest.param(1, 7, 8, id="1 x (7+1) - single consumer, multi-buffer"),
        pytest.param(2, 7, 16, id="2 x (7+1) - multi-consumer, multi-buffer"),
        pytest.param(4, 4, 20, id="4 x (4+1) - multi-consumer, multi-buffer (odd #)"),
        pytest.param(9, 0, 9, id="13 x (0+1) - multi-consumer, empty buffer"),
    ],
)
def test_eventpublisher_broadcast_returns_total_sent(
    test_dir: str, num_consumers: int, num_buffered: int, expected_num_sent: int
) -> None:
    """Verify that a successful broadcast returns the total number of events
    sent, including buffered messages.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param num_consumers: the number of consumers to mock setting up prior to send
    :param num_buffered: the number of pre-buffered events to mock up
    :param expected_num_sent: the expected result from calling send
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumers = []
    for i in range(num_consumers):
        consumers.append(storage_path / f"test-consumer-{i}")

    backbone = BackboneFeatureStore(mock_storage)
    backbone.notification_channels = consumers

    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    # mock building up some buffered events
    for i in range(num_buffered):
        event = OnCreateConsumer(storage_path / f"test-consumer-{str(i)}")
        publisher._event_buffer.append(event)

    assert publisher.num_buffered == num_buffered

    # this event will trigger clearing anything already in buffer
    event0 = OnCreateConsumer(storage_path / f"test-consumer-{num_buffered}")

    # num_receivers should contain a number that computes w/all consumers and all events
    num_receivers = publisher.send(event0)

    assert num_receivers == expected_num_sent
    assert publisher.num_discarded == 0


def test_eventpublisher_prune_unused_consumer(test_dir: str) -> None:
    """Verify that any unused consumers are pruned each time a new event is sent

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    backbone = BackboneFeatureStore(mock_storage)

    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    event = OnCreateConsumer(consumer_descriptor)

    # the only registered cnosumer is in the event, expect no pruning
    backbone.notification_channels = (consumer_descriptor,)

    publisher.send(event)
    assert str(consumer_descriptor) in publisher._channel_cache
    assert len(publisher._channel_cache) == 1

    # add a new descriptor for another event...
    consumer_descriptor2 = storage_path / "test-consumer-2"
    # ... and remove the old descriptor from the backbone when it's looked up
    backbone.notification_channels = (consumer_descriptor2,)

    event = OnCreateConsumer(consumer_descriptor2)

    publisher.send(event)

    assert str(consumer_descriptor2) in publisher._channel_cache
    assert str(consumer_descriptor) not in publisher._channel_cache
    assert len(publisher._channel_cache) == 1

    # test multi-consumer pruning by caching some extra channels
    prune0, prune1, prune2 = "abc", "def", "ghi"
    publisher._channel_cache[prune0] = "doesnt-matter-if-it-is-pruned"
    publisher._channel_cache[prune1] = "doesnt-matter-if-it-is-pruned"
    publisher._channel_cache[prune2] = "doesnt-matter-if-it-is-pruned"

    # add in one of our old channels so we prune the above items, send to these
    backbone.notification_channels = (consumer_descriptor, consumer_descriptor2)

    publisher.send(event)

    assert str(consumer_descriptor2) in publisher._channel_cache
    
    # NOTE: we should NOT prune something that isn't used by this message but
    # does appear in `backbone.notification_channels`
    assert str(consumer_descriptor) in publisher._channel_cache

    # confirm all of our items that were not in the notification channels are gone
    for pruned in [prune0, prune1, prune2]:
        assert pruned not in publisher._channel_cache

    # confirm we have only the two expected items in the channel cache
    assert len(publisher._channel_cache) == 2
