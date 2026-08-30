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

"""Tests for sensitive-field detection, masking and tenant permissions."""

from hugegraph_llm.nl2sql.permissions import (
    PermissionGate,
    is_sensitive,
    mask_value,
)


def test_is_sensitive_heuristic():
    assert is_sensitive("user_phone", "用户手机号") is True
    assert is_sensitive("id_card_no", "身份证号") is True
    assert is_sensitive("gmv", "成交总额") is False


def test_is_sensitive_explicit_flag_wins():
    assert is_sensitive("gmv", "", flag=True) is True
    assert is_sensitive("user_phone", "手机", flag=False) is False


def test_mask_value():
    assert mask_value("13800138000") == "138*****000"
    assert mask_value("abc") == "***"
    assert "*" in mask_value("secret-token-123")


class _FakeItem:
    def __init__(self, name, table):
        self.name = name
        self.table = table


def test_gate_allow_all_without_rules():
    gate = PermissionGate("t1")  # no rules -> allow everything
    assert gate.can_access_column("dw.users", "user_phone") is True
    assert gate.enabled is False


def test_gate_column_allowlist():
    rules = {"t1": {"dw.users": ["user_id"]}}
    gate = PermissionGate("t1", rules)
    assert gate.enabled is True
    assert gate.can_access_column("dw.users", "user_id") is True
    assert gate.can_access_column("dw.users", "user_phone") is False
    assert gate.can_access_column("dw.orders", "gmv") is True  # table not listed


def test_gate_table_allowlist():
    rules = {"t1": ["dw.users"]}
    gate = PermissionGate("t1", rules)
    assert gate.can_access_column("dw.users", "user_phone") is True
    assert gate.can_access_column("dw.orders", "gmv") is False


def test_gate_filter_column_items():
    rules = {"t1": {"dw.users": ["user_id"]}}
    gate = PermissionGate("t1", rules)
    items = [_FakeItem("user_id", "dw.users"),
             _FakeItem("user_phone", "dw.users"),
             _FakeItem("gmv", "dw.orders")]
    kept = gate.filter_column_items(items)
    assert [i.name for i in kept] == ["user_id", "gmv"]
