# # BSD 2-Clause License
# #
# # Copyright (c) 2021-2024, Hewlett Packard Enterprise
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# #    list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# #    this list of conditions and the following disclaimer in the documentation
# #    and/or other materials provided with the distribution.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# import os
# import os.path as osp
# import pathlib
# import shutil
# import typing as t

# import pytest

# from smartsim import Experiment
# from smartsim._core.config import CONFIG
# from smartsim._core.config.config import Config
# from smartsim._core.mli.infrastructure.storage.backbonefeaturestore import (
#     BackboneFeatureStore,
#     EventBroadcaster,
#     EventConsumer,
#     OnCreateConsumer,
# )
# from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
#     DragonFeatureStore,
# )
# from smartsim._core.mli.infrastructure.storage.featurestore import ReservedKeys
# from smartsim._core.utils import serialize
# from smartsim.database import Orchestrator
# from smartsim.entity import Model
# from smartsim.error import SmartSimError
# from smartsim.error.errors import SSUnsupportedError
# from smartsim.settings import RunSettings
# from smartsim.status import SmartSimStatus
# from tests.mli.channel import FileSystemCommChannel
# from tests.mli.featurestore import FileSystemFeatureStore, MemoryFeatureStore

# if t.TYPE_CHECKING:
#     import conftest

# dragon = pytest.importorskip("dragon")


# # The tests in this file belong to the slow_tests group
# pytestmark = pytest.mark.slow_tests


# def test_eventconsumer_receive_failure(test_dir: str) -> None:
#     """Verify that an exception during message retrieval

#     :param test_dir: pytest fixture automatically generating unique working
#     directories for individual test outputs"""
#     storage_path = pathlib.Path(test_dir) / "features"
#     storage_path.mkdir(parents=True, exist_ok=True)

#     mock_storage = {}

#     # note: file-system descriptors are just paths
#     target_descriptor = str(storage_path / "test-consumer")

#     backbone = BackboneFeatureStore(mock_storage)
#     publisher = EventBroadcaster(
#         backbone, channel_factory=FileSystemCommChannel.from_descriptor
#     )
#     event = OnCreateConsumer(target_descriptor)
#     backbone.notification_channels = (target_descriptor,)

#     # send a message into the channel
#     num_sent = publisher.send(event)
#     assert num_sent > 0

#     comm_channel = FileSystemCommChannel.from_descriptor(target_descriptor)
#     consumer = EventConsumer(comm_channel, backbone)

#     all_received: t.List[OnCreateConsumer] = consumer.receive()
#     assert len(all_received) == 1

#     # verify we received the same event that was raised
#     assert all_received[0].type == event.type
#     assert all_received[0].descriptor == event.descriptor
