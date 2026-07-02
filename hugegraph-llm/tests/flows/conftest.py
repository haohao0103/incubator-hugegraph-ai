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

"""conftest for flow tests — ensure pyhugegraph is importable without a real server."""

import logging
import sys
from unittest.mock import MagicMock


def _make_mock_logger(*args, **kwargs):
    logger = logging.getLogger(kwargs.get("logger_name", "mock"))
    logger.setLevel(logging.DEBUG)
    return logger


# Always install a clean mock so that other conftests (e.g. tests/operators/conftest.py)
# which may register a minimal MagicMock do not break imports of pyhugegraph.client.
mock_pyhugegraph = MagicMock()
mock_pyhugegraph_utils = MagicMock()
mock_pyhugegraph_utils_log = MagicMock()
mock_pyhugegraph_utils_log.init_logger = _make_mock_logger
mock_pyhugegraph_utils_exceptions = MagicMock()
mock_pyhugegraph_client = MagicMock()
mock_pyhugegraph_client.PyHugeClient = MagicMock

mock_pyhugegraph.utils = mock_pyhugegraph_utils
mock_pyhugegraph.client = mock_pyhugegraph_client

sys.modules["pyhugegraph"] = mock_pyhugegraph
sys.modules["pyhugegraph.utils"] = mock_pyhugegraph_utils
sys.modules["pyhugegraph.utils.log"] = mock_pyhugegraph_utils_log
sys.modules["pyhugegraph.utils.exceptions"] = mock_pyhugegraph_utils_exceptions
sys.modules["pyhugegraph.client"] = mock_pyhugegraph_client
