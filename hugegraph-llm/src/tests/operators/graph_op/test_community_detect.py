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

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

import hugegraph_llm.operators.graph_op.community_detect as cd
from hugegraph_llm.operators.graph_op.community_detect import CommunityDetect

pytestmark = [pytest.mark.unit]


class _FakeTaskCreateRequest:
    """Records constructor kwargs so tests can assert task_type/params."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestCommunityDetectConstants(unittest.TestCase):
    def test_vermeer_port_is_6688_not_pd_port(self):
        # 8688 is the HugeGraph PD port; Vermeer Master HTTP is 6688
        self.assertEqual(cd.DEFAULT_VERMEER_PORT, 6688)

    def test_default_pd_peers(self):
        self.assertEqual(cd.DEFAULT_PD_PEERS, ["127.0.0.1:8686"])

    def test_vermeer_supported_algorithms(self):
        self.assertEqual(
            cd.VERMEER_SUPPORTED_ALGORITHMS,
            {"leiden", "louvain", "wcc", "pagerank", "degree"},
        )

    def test_degree_constant(self):
        self.assertEqual(cd.ALGORITHM_DEGREE, "degree")


class TestCommunityDetectInitResolve(unittest.TestCase):
    def test_init_defaults(self):
        d = CommunityDetect()
        self.assertEqual(d._pd_peers, ["127.0.0.1:8686"])
        self.assertEqual(d._poll_timeout, 120.0)
        self.assertEqual(d._algorithm, cd.ALGORITHM_LOUVAIN)

    def test_init_custom_pd_peers(self):
        d = CommunityDetect(pd_peers=["10.0.0.1:8686", "10.0.0.2:8686"])
        self.assertEqual(d._pd_peers, ["10.0.0.1:8686", "10.0.0.2:8686"])

    def test_resolve_engine_explicit_vermeer_available(self):
        with patch.object(cd, "HAS_VERMEER", True):
            self.assertEqual(CommunityDetect(engine="vermeer")._resolved_engine, "vermeer")

    def test_resolve_engine_explicit_vermeer_missing_no_client(self):
        with patch.object(cd, "HAS_VERMEER", False):
            self.assertEqual(CommunityDetect(engine="vermeer")._resolved_engine, "networkx")

    def test_resolve_engine_explicit_vermeer_missing_with_client(self):
        with patch.object(cd, "HAS_VERMEER", False):
            d = CommunityDetect(engine="vermeer", client=MagicMock())
            self.assertEqual(d._resolved_engine, "computer")

    def test_resolve_engine_auto_vermeer(self):
        with patch.object(cd, "HAS_VERMEER", True):
            self.assertEqual(CommunityDetect()._resolved_engine, "vermeer")

    def test_resolve_engine_auto_computer(self):
        with patch.object(cd, "HAS_VERMEER", False):
            self.assertEqual(CommunityDetect(client=MagicMock())._resolved_engine, "computer")

    def test_resolve_engine_auto_networkx(self):
        with patch.object(cd, "HAS_VERMEER", False):
            self.assertEqual(CommunityDetect()._resolved_engine, "networkx")


class TestCommunityDetectRun(unittest.TestCase):
    def test_run_networkx(self):
        d = CommunityDetect(client=None, algorithm="louvain", min_community_size=1)
        d._resolved_engine = "networkx"
        ctx = {"vertices": [], "edges": []}
        result = d.run(ctx)
        self.assertEqual(result["communities"], [])
        self.assertEqual(result["community_count"], 0)
        # HAS_LEIDEN=True + louvain -> the local Leiden path reports "leiden"
        self.assertEqual(result["engine_used"], "leiden")

    def test_run_unknown_engine(self):
        d = CommunityDetect()
        d._resolved_engine = "none"
        result = d.run({})
        self.assertEqual(result["communities"], [])
        self.assertEqual(result["community_count"], 0)
        self.assertEqual(result["engine_used"], "none")

    def test_run_vermeer(self):
        with patch.object(cd, "HAS_VERMEER", True), patch(
            "hugegraph_llm.operators.graph_op.community_detect.PyVermeerClient",
            create=True,
        ) as mock_cls, patch(
            "hugegraph_llm.operators.graph_op.community_detect.TaskCreateRequest",
            create=True,
        ), patch("hugegraph_llm.operators.graph_op.community_detect.time.sleep"):
            mock_vermeer = MagicMock()
            mock_cls.return_value = mock_vermeer
            mock_vermeer.tasks.create_task.side_effect = [
                MagicMock(task=MagicMock(id=1)),
                MagicMock(task=MagicMock(id=2)),
            ]
            mock_vermeer.tasks.get_task.return_value = MagicMock(
                task=MagicMock(
                    to_dict=MagicMock(
                        return_value={
                            "state": "SUCCESS",
                            "params": {"result": {"0": ["a", "b", "c"]}},
                        }
                    )
                )
            )
            d = CommunityDetect(engine="vermeer", algorithm="louvain", min_community_size=1)
            result = d.run({})
            self.assertEqual(result["engine_used"], "vermeer")
            self.assertEqual(result["community_count"], 1)

    def test_run_computer(self):
        with patch.object(cd, "HAS_VERMEER", False), patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"vertices": [{"id": "a", "community_id": "0"}, {"id": "b", "community_id": "0"}]}
            mock_post.return_value = mock_resp
            d = CommunityDetect(client=MagicMock(), engine="computer", min_community_size=1)
            result = d.run({})
            self.assertEqual(result["engine_used"], "computer")
            self.assertEqual(result["community_count"], 1)


class TestVermeerEngine(unittest.TestCase):
    def _vermeer_ctx(self, algorithm="louvain", **kwargs):
        """Patch the vermeer stack and return (detector, mock_vermeer)."""
        self._patchers = [
            patch.object(cd, "HAS_VERMEER", True),
            patch(
                "hugegraph_llm.operators.graph_op.community_detect.PyVermeerClient",
                create=True,
            ),
            patch(
                "hugegraph_llm.operators.graph_op.community_detect.TaskCreateRequest",
                _FakeTaskCreateRequest,
                create=True,
            ),
            patch("hugegraph_llm.operators.graph_op.community_detect.time.sleep"),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patchers])
        mock_cls = cd.PyVermeerClient
        mock_vermeer = MagicMock()
        mock_cls.return_value = mock_vermeer
        d = CommunityDetect(engine="vermeer", algorithm=algorithm, **kwargs)
        d._resolved_engine = "vermeer"
        return d, mock_vermeer

    def test_unsupported_algorithm_falls_back(self):
        d, _ = self._vermeer_ctx(algorithm="label_propagation")
        ctx = {"vertices": [], "edges": []}
        result = d._run_vermeer(ctx)
        self.assertEqual(result["engine_used"], "networkx")
        self.assertEqual(result["communities"], [])

    def test_load_task_uses_hugegraph_type_and_pd_peers(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.return_value = MagicMock(task=MagicMock(id=1))
        mock_vermeer.tasks.get_task.return_value = MagicMock(
            task=MagicMock(to_dict=MagicMock(return_value={"state": "SUCCESS", "params": {}}))
        )
        d._run_vermeer({})
        _, kwargs = mock_vermeer.tasks.create_task.call_args_list[0]
        request = kwargs["create_task"]
        self.assertEqual(request.task_type, "load")
        self.assertEqual(request.params["load.type"], "hugegraph")
        self.assertEqual(request.params["load.hg_pd_peers"], json.dumps(d._pd_peers))

    def test_compute_task_params(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="pagerank")
        mock_vermeer.tasks.create_task.return_value = MagicMock(task=MagicMock(id=1))
        mock_vermeer.tasks.get_task.return_value = MagicMock(
            task=MagicMock(
                to_dict=MagicMock(
                    return_value={"state": "SUCCESS", "params": {"result": {"a": 0.5}}}
                )
            )
        )
        d._run_vermeer({})
        _, kwargs = mock_vermeer.tasks.create_task.call_args_list[1]
        request = kwargs["create_task"]
        self.assertEqual(request.task_type, "pagerank")
        self.assertEqual(request.params["compute.algorithm"], "pagerank")
        self.assertEqual(request.params["output.type"], "hugegraph")

    def test_load_submit_failure_falls_back(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.side_effect = RuntimeError("vermeer down")
        ctx = {"vertices": [], "edges": []}
        result = d._run_vermeer(ctx)
        self.assertEqual(result["engine_used"], "networkx")

    def test_load_not_success_falls_back(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.return_value = MagicMock(task=MagicMock(id=1))
        mock_vermeer.tasks.get_task.return_value = MagicMock(
            task=MagicMock(to_dict=MagicMock(return_value={"state": "FAILED", "params": {}}))
        )
        ctx = {"vertices": [], "edges": []}
        result = d._run_vermeer(ctx)
        self.assertEqual(result["engine_used"], "networkx")

    def test_compute_submit_failure_falls_back(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.side_effect = [
            MagicMock(task=MagicMock(id=1)),
            RuntimeError("boom"),
        ]
        mock_vermeer.tasks.get_task.return_value = MagicMock(
            task=MagicMock(to_dict=MagicMock(return_value={"state": "SUCCESS", "params": {}}))
        )
        ctx = {"vertices": [], "edges": []}
        result = d._run_vermeer(ctx)
        self.assertEqual(result["engine_used"], "networkx")

    def test_compute_not_success_falls_back(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.return_value = MagicMock(task=MagicMock(id=1))
        mock_vermeer.tasks.get_task.side_effect = [
            MagicMock(task=MagicMock(to_dict=MagicMock(return_value={"state": "SUCCESS", "params": {}}))),
            MagicMock(task=MagicMock(to_dict=MagicMock(return_value={"state": "CANCELLED", "params": {}}))),
        ]
        ctx = {"vertices": [], "edges": []}
        result = d._run_vermeer(ctx)
        self.assertEqual(result["engine_used"], "networkx")

    def test_vermeer_success_parses_communities(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.return_value = MagicMock(task=MagicMock(id=1))
        mock_vermeer.tasks.get_task.return_value = MagicMock(
            task=MagicMock(
                to_dict=MagicMock(
                    return_value={
                        "state": "SUCCESS",
                        "params": {"result": {"0": ["a", "b", "c"]}},
                    }
                )
            )
        )
        result = d._run_vermeer({})
        self.assertEqual(result["engine_used"], "vermeer")
        self.assertEqual(result["community_count"], 1)
        self.assertEqual(result["communities"][0]["vertices"], ["a", "b", "c"])

    def test_submit_task_returns_none_on_error(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.side_effect = RuntimeError("boom")
        self.assertIsNone(d._submit_vermeer_task(mock_vermeer, "load", "g", {}))

    def test_submit_task_returns_id(self):
        d, mock_vermeer = self._vermeer_ctx(algorithm="wcc")
        mock_vermeer.tasks.create_task.return_value = MagicMock(task=MagicMock(id=42))
        self.assertEqual(d._submit_vermeer_task(mock_vermeer, "compute", "g", {}), 42)

    def test_poll_task_reaches_terminal_state(self):
        d, mock_vermeer = self._vermeer_ctx()
        mock_vermeer.tasks.get_task.side_effect = [
            MagicMock(task=MagicMock(to_dict=MagicMock(return_value={"state": "RUNNING"}))),
            MagicMock(
                task=MagicMock(to_dict=MagicMock(return_value={"state": "SUCCESS"}))
            ),
        ]
        data = d._poll_vermeer_task(mock_vermeer, 1)
        self.assertEqual(data["state"], "SUCCESS")

    def test_poll_task_tolerates_poll_errors(self):
        d, mock_vermeer = self._vermeer_ctx()
        mock_vermeer.tasks.get_task.side_effect = [
            RuntimeError("transient"),
            MagicMock(task=MagicMock(to_dict=MagicMock(return_value={"state": "SUCCESS"}))),
        ]
        data = d._poll_vermeer_task(mock_vermeer, 1)
        self.assertEqual(data["state"], "SUCCESS")

    def test_poll_task_timeout_returns_last_state(self):
        d, mock_vermeer = self._vermeer_ctx(poll_interval=0.01, poll_timeout=0.01)
        mock_vermeer.tasks.get_task.return_value = MagicMock(
            task=MagicMock(to_dict=MagicMock(return_value={"state": "RUNNING"}))
        )
        data = d._poll_vermeer_task(mock_vermeer, 1)
        self.assertEqual(data.get("state"), "RUNNING")

    # -- result parsing -----------------------------------------------------

    def test_parse_vermeer_result_dict_format(self):
        data = {"params": {"result": {"0": ["a", "b"], "1": ["c", "d", "e"]}}}
        communities = CommunityDetect._parse_vermeer_result(data)
        self.assertEqual(len(communities), 2)
        self.assertEqual(communities[0]["id"], "0")

    def test_parse_vermeer_result_list_format(self):
        data = {
            "params": {
                "result": [
                    {"vertex_id": "a", "community_id": "0"},
                    {"id": "b", "community": "0"},
                ]
            }
        }
        communities = CommunityDetect._parse_vermeer_result(data)
        self.assertEqual(len(communities), 1)
        self.assertEqual(communities[0]["id"], "C0")
        self.assertEqual(communities[0]["vertices"], ["a", "b"])

    def test_parse_vermeer_result_filters_small(self):
        data = {"params": {"result": {"0": ["a", "b"], "1": ["only_one"]}}}
        communities = CommunityDetect._parse_vermeer_result(data)
        self.assertEqual(len(communities), 1)

    def test_parse_vermeer_result_empty(self):
        self.assertEqual(CommunityDetect._parse_vermeer_result({}), [])

    def test_parse_vermeer_result_other_type_empty(self):
        self.assertEqual(
            CommunityDetect._parse_vermeer_result({"params": {"result": "junk"}}), []
        )

    def test_parse_vermeer_result_skips_items_without_id(self):
        data = {
            "params": {
                "result": [
                    {"vertex_id": "a", "community_id": "0"},
                    {"vertex_id": "b", "community_id": "0"},
                    {"no_vid": True, "community": "0"},
                ]
            }
        }
        communities = CommunityDetect._parse_vermeer_result(data)
        self.assertEqual(communities[0]["vertices"], ["a", "b"])


class TestComputerEngine(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()

    def _detector(self, **kwargs):
        with patch.object(cd, "HAS_VERMEER", False):
            d = CommunityDetect(client=self.client, engine="computer", **kwargs)
        d._resolved_engine = "computer"
        return d

    @patch("requests.post")
    def test_computer_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vertices": [{"id": "a", "community_id": "0"}, {"id": "b", "community_id": "0"}]}
        mock_post.return_value = mock_resp
        d = self._detector(min_community_size=1)
        result = d._run_computer({})
        self.assertEqual(result["engine_used"], "computer")
        self.assertEqual(result["community_count"], 1)

    @patch("requests.post")
    def test_computer_non_200_falls_back(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "boom"
        mock_post.return_value = mock_resp
        d = self._detector(algorithm="wcc")
        ctx = {"vertices": [], "edges": []}
        result = d._run_computer(ctx)
        self.assertEqual(result["engine_used"], "networkx")

    @patch("requests.post")
    def test_computer_exception_falls_back(self, mock_post):
        mock_post.side_effect = RuntimeError("no computer")
        d = self._detector(algorithm="wcc")
        ctx = {"vertices": [], "edges": []}
        result = d._run_computer(ctx)
        self.assertEqual(result["engine_used"], "networkx")

    def test_parse_computer_result_list(self):
        data = {"vertices": [{"id": "a", "community_id": "0"}, {"vertex_id": "b", "cluster": "0"}]}
        communities = CommunityDetect._parse_computer_result(data)
        self.assertEqual(len(communities), 1)
        self.assertEqual(communities[0]["vertices"], ["a", "b"])

    def test_parse_computer_result_dict(self):
        data = {"result": {"0": ["a", "b"]}}
        communities = CommunityDetect._parse_computer_result(data)
        self.assertEqual(len(communities), 1)
        self.assertEqual(communities[0]["id"], "0")

    def test_parse_computer_result_filters_small_and_empty(self):
        data = {"vertices": [{"id": "a", "community_id": "0"}]}
        self.assertEqual(CommunityDetect._parse_computer_result(data), [])
        self.assertEqual(CommunityDetect._parse_computer_result({}), [])

    def test_parse_computer_result_skips_items_without_id(self):
        data = {
            "vertices": [
                {"id": "a", "community_id": "0"},
                {"id": "b", "community_id": "0"},
                {"no_id": 1},
            ]
        }
        communities = CommunityDetect._parse_computer_result(data)
        self.assertEqual(communities[0]["vertices"], ["a", "b"])

    def test_parse_computer_result_other_type_empty(self):
        self.assertEqual(CommunityDetect._parse_computer_result({"vertices": "junk"}), [])

    def test_parse_computer_result_dict_skips_non_list_values(self):
        data = {"result": {"0": ["a", "b"], "1": "not-a-list"}}
        communities = CommunityDetect._parse_computer_result(data)
        self.assertEqual(len(communities), 1)


class TestNetworkxLeiden(unittest.TestCase):
    """Real igraph/leidenalg path (both installed in the test venv)."""

    def _tri_vertices_edges(self):
        vertices = [
            {"id": "a", "label": "A", "props": {}},
            {"id": "b", "label": "B", "props": {}},
            {"id": "c", "label": "C", "props": {}},
        ]
        edges = [
            {"outV": "a", "inV": "b", "weight": 1, "label": "e1"},
            {"outV": "b", "inV": "c", "weight": 1, "label": "e2"},
            {"outV": "a", "inV": "c", "weight": 1, "label": "e3"},
        ]
        return vertices, edges

    def test_run_networkx_prefers_leiden(self):
        if not cd.HAS_LEIDEN:
            self.skipTest("leidenalg not installed")
        d = CommunityDetect(client=None, algorithm="leiden", min_community_size=1)
        vertices, edges = self._tri_vertices_edges()
        result = d._run_networkx({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "leiden")
        self.assertGreater(result["community_count"], 0)
        comm = result["communities"][0]
        self.assertEqual(comm["modularity_class"], "leiden")
        self.assertGreaterEqual(comm["size"], 1)

    def test_run_networkx_falls_to_louvain_for_other_algorithm(self):
        d = CommunityDetect(client=None, algorithm="wcc", min_community_size=1)
        vertices, edges = self._tri_vertices_edges()
        result = d._run_networkx({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "networkx")

    def test_run_leiden_empty_vertices(self):
        d = CommunityDetect(client=None, algorithm="leiden")
        result = d._run_leiden({"vertices": [], "edges": []})
        self.assertEqual(result["communities"], [])
        self.assertEqual(result["engine_used"], "leiden")

    @patch("hugegraph_llm.operators.graph_op.community_detect.leidenalg")
    def test_run_leiden_partition_failure_falls_back_to_louvain(self, mock_leidenalg):
        mock_leidenalg.find_partition.side_effect = [TypeError, RuntimeError]
        mock_leidenalg.ModularityVertexPartition = MagicMock()
        mock_leidenalg.CPMVertexPartition = MagicMock()
        d = CommunityDetect(client=None, algorithm="leiden", min_community_size=1)
        vertices, edges = self._tri_vertices_edges()
        result = d._run_leiden({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "networkx")

    def test_run_leiden_skips_invalid_vertices_and_edges(self):
        d = CommunityDetect(client=None, algorithm="leiden", min_community_size=1)
        vertices = [
            {"id": "a", "label": "A", "props": {}},
            {"id": "b", "label": "B", "props": {}},
            {"label": "no-id"},  # skipped (no id)
        ]
        edges = [
            {"outV": "a", "inV": "b", "weight": 1, "label": "e1"},
            {"outV": "x", "inV": "y", "weight": 1, "label": "ghost"},  # endpoints unknown
            {"outV": "", "inV": "a", "weight": 1, "label": "empty"},  # empty endpoint
        ]
        result = d._run_leiden({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "leiden")

    def test_run_leiden_no_valid_edges(self):
        d = CommunityDetect(client=None, algorithm="leiden", min_community_size=1)
        vertices = [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]
        edges = [{"outV": "x", "inV": "y", "weight": 1}]  # no valid edge
        result = d._run_leiden({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "leiden")

    def test_run_leiden_filters_small_communities(self):
        d = CommunityDetect(client=None, algorithm="leiden", min_community_size=10)
        vertices = [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]
        edges = [{"outV": "a", "inV": "b", "weight": 1}]
        result = d._run_leiden({"vertices": vertices, "edges": edges})
        self.assertEqual(result["communities"], [])

    def test_run_leiden_fetches_graph_when_missing(self):
        d = CommunityDetect(client=MagicMock(), algorithm="leiden", min_community_size=1)
        vertices, edges = self._tri_vertices_edges()
        d._fetch_graph_from_hugegraph = MagicMock(return_value=(vertices, edges))
        result = d._run_leiden({"vertices": None, "edges": None})
        self.assertEqual(result["engine_used"], "leiden")
        self.assertGreater(result["community_count"], 0)


class TestNetworkxLouvain(unittest.TestCase):
    def _tri_vertices_edges(self):
        vertices = [
            {"id": "a", "label": "A", "props": {}},
            {"id": "b", "label": "B", "props": {}},
            {"id": "c", "label": "C", "props": {}},
        ]
        edges = [
            {"outV": "a", "inV": "b", "weight": 1, "label": "e1"},
            {"outV": "b", "inV": "c", "weight": 1, "label": "e2"},
            {"outV": "a", "inV": "c", "weight": 1, "label": "e3"},
        ]
        return vertices, edges

    def test_run_louvain_with_context(self):
        d = CommunityDetect(client=None, algorithm="louvain", min_community_size=1)
        vertices, edges = self._tri_vertices_edges()
        result = d._run_louvain({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "networkx")
        self.assertGreaterEqual(result["community_count"], 0)
        for comm in result["communities"]:
            self.assertIn("density", comm)

    def test_run_louvain_empty_vertices(self):
        d = CommunityDetect(client=None, algorithm="louvain")
        result = d._run_louvain({"vertices": [], "edges": []})
        self.assertEqual(result["communities"], [])
        self.assertEqual(result["engine_used"], "networkx")

    def test_run_louvain_fetches_graph_when_missing(self):
        d = CommunityDetect(client=MagicMock(), algorithm="louvain", min_community_size=1)
        vertices, edges = self._tri_vertices_edges()
        d._fetch_graph_from_hugegraph = MagicMock(return_value=(vertices, edges))
        result = d._run_louvain({"vertices": None, "edges": None})
        self.assertEqual(result["engine_used"], "networkx")

    def test_run_louvain_skips_invalid_vertices_and_edges(self):
        d = CommunityDetect(client=None, algorithm="louvain", min_community_size=1)
        vertices = [
            {"id": "a", "label": "A", "props": {}},
            {"label": "no-id"},  # skipped
        ]
        edges = [
            {"outV": "a", "inV": "b", "weight": 1, "label": "e1"},
            {"outV": "", "inV": "a", "weight": 1, "label": "empty"},  # empty endpoint
        ]
        result = d._run_louvain({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "networkx")

    def test_run_louvain_filters_small_communities(self):
        d = CommunityDetect(client=None, algorithm="louvain", min_community_size=10)
        vertices = [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]
        edges = [{"outV": "a", "inV": "b", "weight": 1}]
        result = d._run_louvain({"vertices": vertices, "edges": edges})
        self.assertEqual(result["communities"], [])

    def test_run_louvain_dedupes_parallel_edges(self):
        d = CommunityDetect(client=None, algorithm="louvain", min_community_size=1)
        vertices = [{"id": "a", "label": "A", "props": {}}, {"id": "b", "label": "B", "props": {}}]
        edges = [
            {"outV": "a", "inV": "b", "weight": 1, "label": "e1"},
            {"outV": "a", "inV": "b", "weight": 1, "label": "e2"},
        ]
        result = d._run_louvain({"vertices": vertices, "edges": edges})
        self.assertEqual(result["engine_used"], "networkx")

    def test_fallback_networkx(self):
        d = CommunityDetect(client=None, algorithm="wcc", min_community_size=1)
        d._fetch_graph_from_hugegraph = MagicMock(return_value=([], []))
        result = d._fallback_networkx({"vertices": None, "edges": None})
        self.assertEqual(result["engine_used"], "networkx")
        self.assertEqual(d._resolved_engine, "networkx")


class TestFetchGraph(unittest.TestCase):
    def test_fetch_graph_success(self):
        client = MagicMock()
        client.gremlin().exec.return_value = {
            "data": [
                {
                    "vertices": [{"id": "a"}],
                    "edges": [{"outV": "a", "inV": "b"}],
                }
            ]
        }
        d = CommunityDetect(client=client)
        vertices, edges = d._fetch_graph_from_hugegraph()
        self.assertEqual(vertices, [{"id": "a"}])
        self.assertEqual(edges, [{"outV": "a", "inV": "b"}])

    def test_fetch_graph_failure_returns_empty(self):
        client = MagicMock()
        client.gremlin().exec.side_effect = RuntimeError("boom")
        d = CommunityDetect(client=client)
        self.assertEqual(d._fetch_graph_from_hugegraph(), ([], []))


if __name__ == "__main__":
    unittest.main()
