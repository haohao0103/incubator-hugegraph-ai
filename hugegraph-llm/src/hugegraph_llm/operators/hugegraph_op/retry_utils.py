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
"""Connection-fault retry helper.

Generalized from the NebulaGraphStore's tenacity-based SessionPool pattern:
transient connection errors (network hiccups, server restarts) are retried
with exponential backoff, while business errors propagate immediately.

Usage::

    from hugegraph_llm.operators.hugegraph_op.retry_utils import (
        retry_on_connection_error,
    )

    class Foo:
        @retry_on_connection_error(max_attempts=3)
        def read(self) -> ...:
            return self.client.schema().getSchema()
"""

from __future__ import annotations

from typing import Any, Callable

from pyhugegraph.utils.exceptions import ServerError
from requests.exceptions import RequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Errors considered transient: transport/connection-level failures where a
# retry may succeed once the server recovers.
TRANSIENT_ERRORS = (RequestException, ServerError, ConnectionError, TimeoutError)


def retry_on_connection_error(
    max_attempts: int = 3,
    base: float = 0.5,
    max_wait: float = 4.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory: retry transient errors with exponential backoff.

    Args:
        max_attempts: Total attempts (including the first call).
        base: Exponential backoff multiplier (seconds).
        max_wait: Upper bound for the backoff delay (seconds).
    """
    return retry(
        retry=retry_if_exception_type(TRANSIENT_ERRORS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base, max=max_wait),
        reraise=True,
    )
