# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
import typing as t
import uuid

import pytest

from smartsim._core.entrypoints.telemetrymonitor import (
    DbConnectionCollector,
    DbMemoryCollector,
    JobEntity,
    redis,
)
from smartsim.error import SmartSimError

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.group_a


@pytest.fixture
def mock_con():
    def _mock_con(min=1, max=1000):
        i = min
        while True:
            yield [{"addr": f"127.0.0.{i}:1234"}, {"addr": f"127.0.0.{i}:2345"}]
            i += 1
            if i > max:
                return None

    return _mock_con


@pytest.fixture
def mock_mem():
    def _mock_mem(min=1, max=1000):
        i = min
        while True:
            yield {
                "total_system_memory": 1000 * i,
                "used_memory": 1111 * i,
                "used_memory_peak": 1234 * i,
            }
            i += 1
            if i > max:
                return None

    return _mock_mem


@pytest.fixture
def mock_redis():
    def _mock_redis(
        is_conn: bool = True,
        conn_side_effect=None,
        mem_stats=None,
        client_stats=None,
        coll_side_effect=None,
    ):
        class MockConn:
            def __init__(self, *args, **kwargs) -> None:
                if conn_side_effect is not None:
                    conn_side_effect()

            async def info(self) -> t.Dict[str, t.Any]:
                if coll_side_effect:
                    await coll_side_effect()

                if mem_stats:
                    return next(mem_stats)
                return {
                    "ts": 111,
                    "total_system_memory": "111",
                    "used_memory": "222",
                    "used_memory_peak": "333",
                }

            async def client_list(self) -> t.Dict[str, t.Any]:
                if coll_side_effect:
                    await coll_side_effect()

                if client_stats:
                    return next(client_stats)
                return {"ts": 111, "addr": "127.0.0.1"}

        return MockConn

    return _mock_redis


@pytest.fixture
def mock_entity(test_dir):
    def _mock_entity(
        host: str = "127.0.0.1", port: str = "6379", name: str = "", type: str = ""
    ):
        entity = JobEntity()
        entity.name = name if name else str(uuid.uuid4())
        entity.status_dir = test_dir
        entity.type = type
        entity.meta = {
            "host": host,
            "port": port,
        }
        return entity

    return _mock_entity


@pytest.mark.asyncio
async def test_dbmemcollector_prepare(mock_entity, mock_sink):
    """Ensure that collector preparation succeeds when expected"""
    entity = mock_entity()

    collector = DbMemoryCollector(entity, mock_sink())
    await collector.prepare()
    assert collector._client


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail(
    mock_entity, mock_sink, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that collector preparation reports a failure to connect"""
    entity = mock_entity()

    with monkeypatch.context() as ctx:
        # mock up a redis constructor that returns None
        ctx.setattr(redis, "Redis", lambda host, port: None)

        with pytest.raises(SmartSimError) as ex:
            collector = DbMemoryCollector(entity, mock_sink())
            await collector.prepare()

        assert not collector._client

        err_content = ",".join(ex.value.args)
        assert "connect" in err_content


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail_dep(
    mock_entity, mock_sink, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that collector preparation attempts to connect, ensure it
    reports a failure if the db conn bombs"""
    entity = mock_entity()

    def raiser():
        # mock raising exception on connect attempts to test err handling
        raise redis.ConnectionError("mock connection failure")

    collector = DbMemoryCollector(entity, mock_sink())
    with monkeypatch.context() as ctx:
        ctx.setattr(redis, "Redis", raiser)
        with pytest.raises(SmartSimError) as ex:
            await collector.prepare()

        assert not collector._client

        err_content = ",".join(ex.value.args)
        assert "communicate" in err_content


@pytest.mark.asyncio
async def test_dbmemcollector_collect(
    mock_entity, mock_redis, mock_mem, mock_sink, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    sink = mock_sink()
    collector = DbMemoryCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(redis, "Redis", mock_redis(mem_stats=mock_mem(1, 2)))

        await collector.prepare()
        await collector.collect()
        stats = collector.value

        assert set(
            ("ts", "total_system_memory", "used_memory", "used_memory_peak")
        ) == set(sink.args)
        assert set((1000, 1111, 1234)).issubset(set(sink.args.values()))


@pytest.mark.asyncio
async def test_dbmemcollector_integration(mock_entity, mock_sink, local_db):
    """Integration test with a real orchestrator instance to ensure
    output data matches expectations and proper db client API uage"""
    entity = mock_entity(port=local_db.ports[0])

    collector = DbMemoryCollector(entity, mock_sink())

    await collector.prepare()
    await collector.collect()
    stats = collector.value

    assert len(stats) == 3  # prove we filtered to expected data size
    assert stats["used_memory"] > 0  # prove used_memory was retrieved
    assert stats["used_memory_peak"] > 0  # prove used_memory_peak was retrieved
    assert stats["total_system_memory"] > 0  # prove total_system_memory was retrieved


@pytest.mark.asyncio
async def test_dbconncollector_collect(
    mock_entity, mock_sink, mock_redis, mock_con, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    collector = DbConnectionCollector(entity, mock_sink())
    with monkeypatch.context() as ctx:
        ctx.setattr(redis, "Redis", mock_redis(client_stats=mock_con(1, 2)))

        await collector.prepare()
        await collector.collect()

        stats = collector.value

        assert set(["127.0.0.1:1234", "127.0.0.1:2345"]) == set(stats)


@pytest.mark.asyncio
async def test_dbconncollector_integration(mock_entity, mock_sink, local_db):
    """Integration test with a real orchestrator instance to ensure
    output data matches expectations and proper db client API uage"""
    entity = mock_entity(port=local_db.ports[0])

    collector = DbConnectionCollector(entity, mock_sink())

    await collector.prepare()
    await collector.collect()
    stats = collector.value

    assert len(stats) == 1
    assert stats[0]
