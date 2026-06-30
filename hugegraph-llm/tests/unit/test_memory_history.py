# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for MemoryHistoryTracker — mem0-style version history."""

import sys
import os
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.memory_history import MemoryHistoryTracker, HistoryEvent


@pytest.fixture
def tracker():
    """Create a MemoryHistoryTracker with a temp database."""
    db_path = tempfile.mktemp(suffix=".db")
    t = MemoryHistoryTracker(db_path=db_path)
    yield t
    t.close()
    # Cleanup: remove singleton reference
    MemoryHistoryTracker._instances.pop(db_path, None)
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ── add_history Tests ─────────────────────────────────────────


class TestAddHistory:

    def test_add_event(self, tracker):
        evt = tracker.add_history(
            memory_id="mem001",
            event="ADD",
            new_memory="张三在货拉拉公司工作",
            actor_id="user1",
            role="user",
        )
        assert evt.memory_id == "mem001"
        assert evt.event == "ADD"
        assert evt.new_memory == "张三在货拉拉公司工作"
        assert evt.actor_id == "user1"

    def test_update_event(self, tracker):
        evt = tracker.add_history(
            memory_id="mem001",
            event="UPDATE",
            old_memory="张三在货拉拉公司工作",
            new_memory="张三已离职",
        )
        assert evt.event == "UPDATE"
        assert evt.old_memory == "张三在货拉拉公司工作"
        assert evt.new_memory == "张三已离职"

    def test_delete_event(self, tracker):
        evt = tracker.add_history(
            memory_id="mem001",
            event="DELETE",
            old_memory="old content",
            is_deleted=True,
        )
        assert evt.event == "DELETE"
        assert evt.is_deleted is True

    def test_with_metadata(self, tracker):
        evt = tracker.add_history(
            memory_id="mem002",
            event="ADD",
            new_memory="test",
            metadata={"source": "chat", "session_id": "s1"},
        )
        assert evt.metadata["source"] == "chat"


# ── batch_add_history Tests ───────────────────────────────────


class TestBatchAddHistory:

    def test_batch_insert(self, tracker):
        records = [
            {"memory_id": "mem1", "event": "ADD", "new_memory": "fact1"},
            {"memory_id": "mem2", "event": "ADD", "new_memory": "fact2"},
            {"memory_id": "mem3", "event": "ADD", "new_memory": "fact3"},
        ]
        tracker.batch_add_history(records)
        assert tracker.count_events() == 3

    def test_batch_with_actor(self, tracker):
        records = [
            {"memory_id": "mem1", "event": "ADD", "new_memory": "f1",
             "actor_id": "user1", "role": "user"},
            {"memory_id": "mem2", "event": "ADD", "new_memory": "f2",
             "actor_id": "user1", "role": "user"},
        ]
        tracker.batch_add_history(records)
        events = tracker.get_history("mem1")
        assert len(events) == 1
        assert events[0].actor_id == "user1"


# ── get_history Tests ─────────────────────────────────────────


class TestGetHistory:

    def test_history_timeline(self, tracker):
        tracker.add_history("mem001", "ADD", new_memory="version 1")
        tracker.add_history("mem001", "UPDATE", old_memory="version 1", new_memory="version 2")
        tracker.add_history("mem001", "UPDATE", old_memory="version 2", new_memory="version 3")

        history = tracker.get_history("mem001")
        assert len(history) == 3
        # Events should be ordered by created_at ascending
        assert history[0].event == "ADD"
        assert history[1].event == "UPDATE"
        assert history[2].event == "UPDATE"
        # Content should track the evolution
        assert history[0].new_memory == "version 1"
        assert history[1].old_memory == "version 1"
        assert history[1].new_memory == "version 2"

    def test_empty_history(self, tracker):
        history = tracker.get_history("nonexistent")
        assert history == []


# ── get_recent_history Tests ──────────────────────────────────


class TestGetRecentHistory:

    def test_recent_all(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem2", "ADD", new_memory="f2")
        recent = tracker.get_recent_history(limit=10)
        assert len(recent) == 2

    def test_recent_by_type(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem1", "UPDATE", old_memory="f1", new_memory="f1_updated")
        adds = tracker.get_recent_history(limit=10, event_type="ADD")
        assert len(adds) == 1
        assert adds[0].event == "ADD"


# ── count_events Tests ────────────────────────────────────────


class TestCountEvents:

    def test_count_all(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem1", "UPDATE", old_memory="f1", new_memory="f2")
        tracker.add_history("mem2", "ADD", new_memory="f3")
        assert tracker.count_events() == 3

    def test_count_by_memory(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem2", "ADD", new_memory="f2")
        assert tracker.count_events(memory_id="mem1") == 1

    def test_count_by_type(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem1", "UPDATE", old_memory="f1", new_memory="f2")
        assert tracker.count_events(event_type="ADD") == 1


# ── get_stats Tests ───────────────────────────────────────────


class TestGetStats:

    def test_stats(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem1", "UPDATE", old_memory="f1", new_memory="f2")
        tracker.add_history("mem2", "ADD", new_memory="f3")
        tracker.add_history("mem3", "DELETE", old_memory="f4", is_deleted=True)

        stats = tracker.get_stats()
        assert stats["total_events"] == 4
        assert stats["add_events"] == 2
        assert stats["update_events"] == 1
        assert stats["delete_events"] == 1
        assert stats["unique_memories"] == 3

    def test_empty_stats(self, tracker):
        stats = tracker.get_stats()
        assert stats["total_events"] == 0
        assert stats["unique_memories"] == 0


# ── clear Tests ───────────────────────────────────────────────


class TestClear:

    def test_clear(self, tracker):
        tracker.add_history("mem1", "ADD", new_memory="f1")
        tracker.add_history("mem2", "ADD", new_memory="f2")
        count = tracker.clear()
        assert count == 2
        assert tracker.count_events() == 0


# ── Singleton Tests ───────────────────────────────────────────


class TestSingleton:

    def test_same_path_same_instance(self):
        db_path = tempfile.mktemp(suffix=".db")
        t1 = MemoryHistoryTracker(db_path=db_path)
        t2 = MemoryHistoryTracker(db_path=db_path)
        assert t1 is t2
        t1.close()
        MemoryHistoryTracker._instances.pop(db_path, None)

    def test_different_path_different_instance(self):
        p1 = tempfile.mktemp(suffix=".db1")
        p2 = tempfile.mktemp(suffix=".db2")
        t1 = MemoryHistoryTracker(db_path=p1)
        t2 = MemoryHistoryTracker(db_path=p2)
        assert t1 is not t2
        t1.close()
        t2.close()
        MemoryHistoryTracker._instances.pop(p1, None)
        MemoryHistoryTracker._instances.pop(p2, None)


# ── HistoryEvent Dataclass Tests ──────────────────────────────


class TestHistoryEvent:

    def test_create_event(self):
        evt = HistoryEvent(
            id="abc123",
            memory_id="mem001",
            event="ADD",
            new_memory="test content",
            created_at=1234567890.0,
        )
        assert evt.id == "abc123"
        assert evt.event == "ADD"
        assert evt.is_deleted is False
