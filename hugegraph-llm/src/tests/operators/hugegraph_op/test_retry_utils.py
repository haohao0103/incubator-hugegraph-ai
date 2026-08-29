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

import unittest

import pytest
from pyhugegraph.utils.exceptions import ServerError
from requests.exceptions import RequestException

from hugegraph_llm.operators.hugegraph_op.retry_utils import (
    TRANSIENT_ERRORS,
    retry_on_connection_error,
)

pytestmark = [pytest.mark.unit]


class TestRetryOnConnectionError(unittest.TestCase):
    def test_transient_error_types(self):
        self.assertIn(RequestException, TRANSIENT_ERRORS)
        self.assertIn(ServerError, TRANSIENT_ERRORS)
        self.assertIn(ConnectionError, TRANSIENT_ERRORS)

    def test_succeeds_on_first_try(self):
        calls = []

        @retry_on_connection_error(max_attempts=3, base=0.01)
        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_transient_then_succeeds(self):
        calls = []

        @retry_on_connection_error(max_attempts=3, base=0.01)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise RequestException("transient")
            return "recovered"

        self.assertEqual(fn(), "recovered")
        self.assertEqual(len(calls), 3)

    def test_exhausts_attempts_and_reraises(self):
        calls = []

        @retry_on_connection_error(max_attempts=2, base=0.01)
        def fn():
            calls.append(1)
            raise ServerError("down")

        with self.assertRaises(ServerError):
            fn()
        self.assertEqual(len(calls), 2)

    def test_business_error_not_retried(self):
        calls = []

        @retry_on_connection_error(max_attempts=3, base=0.01)
        def fn():
            calls.append(1)
            raise ValueError("business error")

        with self.assertRaises(ValueError):
            fn()
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
