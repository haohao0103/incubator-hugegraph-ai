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

"""conftest for EDC+Guided schema tests — mock pyhugegraph.utils.log dependency."""

import logging
import sys
from unittest.mock import MagicMock

# Create a mock pyhugegraph package hierarchy before any hugegraph_llm imports
mock_pyhugegraph = MagicMock()
mock_pyhugegraph_utils = MagicMock()
mock_pyhugegraph_utils_log = MagicMock()

# init_logger should return a standard Python logger
def mock_init_logger(*args, **kwargs):
    logger = logging.getLogger(kwargs.get("logger_name", "mock"))
    logger.setLevel(logging.DEBUG)
    return logger

mock_pyhugegraph_utils_log.init_logger = mock_init_logger

# Register the mock modules in sys.modules so imports work
sys.modules["pyhugegraph"] = mock_pyhugegraph
sys.modules["pyhugegraph.utils"] = mock_pyhugegraph_utils
sys.modules["pyhugegraph.utils.log"] = mock_pyhugegraph_utils_log

# Mock log module should also provide a pre-configured logger
# This is used by hugegraph_llm.utils.log which imports init_logger
mock_pyhugegraph.utils.log.init_logger = mock_init_logger
