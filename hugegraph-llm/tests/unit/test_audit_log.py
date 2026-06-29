# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for hugegraph_llm.utils.audit_log."""

import json
import os
import tempfile

import pytest

from hugegraph_llm.utils.audit_log import AuditEvent, AuditLogger, get_audit_logger


@pytest.fixture
def temp_logger():
    """Provide an isolated in-memory audit logger."""
    old_instances = dict(AuditLogger._instances)
    AuditLogger._instances.clear()
    logger = AuditLogger(db_path=":memory:")
    yield logger
    AuditLogger._instances.clear()
    AuditLogger._instances.update(old_instances)


def test_log_event(temp_logger):
    event = temp_logger.log(
        operation="add_memory",
        user_id="u1",
        memory_id="m1",
        content="hello",
        latency_ms=12.5,
        success=True,
        metadata={"k": "v"},
    )
    assert isinstance(event, AuditEvent)
    assert event.operation == "add_memory"
    assert event.user_id == "u1"
    assert event.memory_id == "m1"
    assert event.content == "hello"
    assert event.latency_ms == 12.5
    assert event.success is True
    assert json.loads(event.metadata) == {"k": "v"}


def test_get_events(temp_logger):
    temp_logger.log(operation="add_memory", user_id="u1")
    temp_logger.log(operation="search_memory", user_id="u1")
    temp_logger.log(operation="add_memory", user_id="u2")

    all_events = temp_logger.get_events()
    assert len(all_events) == 3

    u1_events = temp_logger.get_events(user_id="u1")
    assert len(u1_events) == 2

    add_events = temp_logger.get_events(operation="add_memory")
    assert len(add_events) == 2


def test_count_and_stats(temp_logger):
    temp_logger.log(operation="add_memory", user_id="u1", latency_ms=10, success=True)
    temp_logger.log(operation="search_memory", user_id="u1", latency_ms=20, success=True)
    temp_logger.log(operation="add_memory", user_id="u2", latency_ms=30, success=False)

    assert temp_logger.count() == 3
    assert temp_logger.count(user_id="u1") == 2
    assert temp_logger.count(operation="add_memory") == 2

    stats = temp_logger.get_stats()
    assert stats["total_events"] == 3
    assert stats["successful_events"] == 2
    assert stats["failed_events"] == 1
    assert stats["avg_latency_ms"] == 20.0
    assert stats["operations"]["add_memory"] == 2
    assert stats["operations"]["search_memory"] == 1


def test_clear(temp_logger):
    temp_logger.log(operation="add_memory", user_id="u1")
    assert temp_logger.count() == 1
    deleted = temp_logger.clear()
    assert deleted == 1
    assert temp_logger.count() == 0


def test_singleton_and_env_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "audit.db")
        os.environ["MEMORY_AUDIT_LOG_PATH"] = db_path
        logger = get_audit_logger()
        assert logger._db_path == db_path
        logger.log(operation="add_memory", user_id="x")
        assert logger.count() == 1
        del os.environ["MEMORY_AUDIT_LOG_PATH"]
        # Clean singleton to avoid cross-test leakage
        AuditLogger._instances.pop(logger._db_path, None)


def test_log_failure_still_records(temp_logger):
    event = temp_logger.log(
        operation="search_memory",
        user_id="u1",
        success=False,
        error="timeout",
    )
    assert event.success is False
    assert event.error == "timeout"
    assert temp_logger.count() == 1
