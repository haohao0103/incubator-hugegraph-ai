# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the two REST-encoding bugs found on a live HugeGraph.

Both were invisible to the driver's own (offline) test suite because that
suite uses a fake client; they only surface against a real server.
"""

import pytest

from hugegraph_llm.memory.driver import HugeGraphClient, _json_str


# -- value encoding ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1704067200000, 1704067200000),   # epoch millis must stay numeric
        (42, 42),
        (3.14, 3.14),
        (True, True),
        (False, False),
    ],
)
def test_numeric_and_boolean_values_pass_through(value, expected):
    """Stringifying them makes the server reject LONG/DOUBLE/BOOLEAN keys."""
    assert _json_str(value) == expected
    assert not isinstance(_json_str(value), str) or isinstance(value, str)


def test_string_values_stay_strings():
    assert _json_str("2024-01-01") == "2024-01-01"


def test_list_and_dict_are_json_encoded():
    assert _json_str(["a", "b"]) == '["a", "b"]'
    assert _json_str({"k": 1}) == '{"k": 1}'


def test_none_stays_none():
    assert _json_str(None) is None


def test_epoch_millis_round_trips_as_long():
    """The exact failure: '1704067200000' rejected for a Long property."""
    encoded = _json_str(1704067200000)
    assert isinstance(encoded, int)
    assert encoded == 1704067200000


# -- id quoting -------------------------------------------------------------


def test_vertex_id_is_quoted():
    assert HugeGraphClient._qid("u1") == '"u1"'


def test_edge_id_is_not_quoted():
    """Quoting an edge id makes HugeGraph reject it: "Invalid format"."""

    class Capturing(HugeGraphClient):
        def __init__(self):
            self.paths = []

        def _req(self, method, path, **kw):
            self.paths.append(path)
            return {}

    client = Capturing()
    # Edge ids contain '>' -- quoting them breaks the request.
    client.update_edge("Su1>1>1>>Sc1", "RELATES_TO", {"expired_at": 1})
    assert client.paths == ['/graph/edges/Su1>1>1>>Sc1?action=append']
    assert '"' not in client.paths[0]
