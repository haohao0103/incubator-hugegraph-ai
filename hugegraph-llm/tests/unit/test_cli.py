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

"""Tests for hugegraph_llm.cli."""

import json
from unittest import mock

import pytest

from hugegraph_llm import cli


@pytest.fixture
def mock_backend():
    """Patch the CLI backend factory and return a MagicMock backend."""
    with mock.patch.object(cli, "_get_backend") as mock_factory:
        backend = mock_factory.return_value
        yield backend


def test_main_no_command(capsys):
    assert cli.main([]) == 1
    captured = capsys.readouterr()
    assert "usage" in captured.out


def test_cmd_add_success(mock_backend, capsys):
    mock_backend.add_memory.return_value = {"memory_id": "abc", "action": "ADD"}
    assert cli.main(["add", "hello"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["memory_id"] == "abc"


def test_cmd_add_failure(mock_backend, capsys):
    mock_backend.add_memory.return_value = {"error": "bad"}
    assert cli.main(["add", "hello"]) == 1


def test_cmd_search_success(mock_backend, capsys):
    mock_backend.search_memory.return_value = {"answer": "yes", "results": []}
    assert cli.main(["search", "where"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["answer"] == "yes"


def test_cmd_search_error(mock_backend, capsys):
    mock_backend.search_memory.return_value = {"error": "not found"}
    assert cli.main(["search", "where"]) == 1


def test_cmd_list(mock_backend, capsys):
    mock_backend.list_memories.return_value = [{"id": "1", "content": "a"}]
    assert cli.main(["list"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert len(output) == 1


def test_cmd_delete_success(mock_backend, capsys):
    mock_backend.delete_memory.return_value = {"status": "ok"}
    assert cli.main(["delete", "m1"]) == 0


def test_cmd_delete_failure(mock_backend, capsys):
    mock_backend.delete_memory.return_value = {"error": "NOT_FOUND"}
    assert cli.main(["delete", "m1"]) == 1


def test_cmd_update_success(mock_backend, capsys):
    mock_backend.update_memory.return_value = {"status": "ok"}
    assert cli.main(["update", "m1", "new"]) == 0


def test_cmd_stats(mock_backend, capsys):
    mock_backend.get_stats.return_value = {"memories": 1}
    assert cli.main(["stats"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["memories"] == 1


def test_cmd_rewrite(mock_backend, capsys):
    mock_backend.rewrite_query.return_value = {
        "original": "test",
        "rewritten": "test",
        "variants": ["test"],
    }
    assert cli.main(["rewrite", "test"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["original"] == "test"


def test_cmd_audit_stats(capsys):
    assert cli.main(["audit", "--stats"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert "total_events" in output


def test_cmd_audit_events(capsys):
    assert cli.main(["audit", "--limit", "5"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert isinstance(output, list)


def test_cmd_locomo(mock_backend, tmp_path, capsys):
    result_file = tmp_path / "locomo_result.json"
    script_file = tmp_path / "locomo.py"
    script_file.write_text(
        "def run_locomo(*args, **kwargs):\n"
        "    return {\n"
        "        'metrics': {\n"
        "            'sessions': 1, 'total_questions': 2, 'correct': 1,\n"
        "            'accuracy': 0.5, 'hit_at_5': 1.0, 'mrr': 0.75,\n"
        "            'avg_latency_ms': 100, 'p95_latency_ms': 200, 'token_estimate': 50,\n"
        "        },\n"
        "        'details': [],\n"
        "    }\n"
    )
    assert cli.main(["locomo", "--output", str(result_file), "--script-path", str(script_file)]) == 0
    captured = capsys.readouterr()
    assert "LOCOMO Benchmark Results" in captured.out
    assert result_file.exists()


def test_cmd_server(mock_backend):
    server_main = mock.MagicMock()
    with mock.patch.object(cli, "_load_server_main", return_value=server_main):
        assert cli.main(["server", "--port", "9999"]) == 0
    server_main.assert_called_once()


def test_build_parser_top_k():
    parser = cli._build_parser()
    args = parser.parse_args(["search", "q", "--top-k", "3"])
    assert args.top_k == 3
