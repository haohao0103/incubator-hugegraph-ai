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

"""Sensitive-field detection, masking and tenant-level column permissions.

Mechanism first: no permission rules means *everyone can see everything*
(backward compatible). When rules are configured they are enforced:

    rules = {
        "tenant_a": {"dw.users": ["user_id", "city"]},   # column allow-list
        "tenant_b": ["dw.users"],                        # whole-table allow-list
    }

``PermissionGate(tenant, rules).can_access_column(table, col)`` returns False
for columns not granted to the tenant. Sensitive detection is heuristic
(name/comment patterns); governance can override via the metadata
``sensitive`` flag on a column.
"""

import re
from typing import Dict, List, Optional, Union

SENSITIVE_PATTERNS = (
    "phone", "mobile", "tel", "手机", "电话", "身份证", "id_card",
    "password", "passwd", "pwd", "token", "secret", "银行卡", "bank_card",
    "account_no", "账号", "邮箱", "email",
)

_SENSITIVE_RE = re.compile("|".join(re.escape(p) for p in SENSITIVE_PATTERNS),
                           re.IGNORECASE)


def is_sensitive(name: str, comment: str = "", flag: Optional[bool] = None) -> bool:
    """Heuristic sensitive detection; an explicit metadata flag wins."""
    if flag is not None:
        return bool(flag)
    text = f"{name} {comment}"
    return bool(_SENSITIVE_RE.search(text))


def mask_value(value: str) -> str:
    """Generic masking: keep first and last char(s), mask the middle."""
    v = str(value)
    if len(v) <= 4:
        return "*" * len(v)
    if len(v) <= 8:
        return v[:2] + "*" * (len(v) - 4) + v[-2:]
    return v[:3] + "*" * (len(v) - 6) + v[-3:]


class PermissionGate:
    """Tenant-level column access control (empty rules = allow all)."""

    def __init__(self, tenant: Optional[str] = None,
                 rules: Optional[Dict[str, Union[List[str], Dict[str, List[str]]]]] = None):
        self._tenant = tenant or ""
        self._rules = rules or {}

    @property
    def enabled(self) -> bool:
        return bool(self._rules and self._tenant in self._rules)

    def can_access_column(self, table: str, column: str) -> bool:
        if not self.enabled:
            return True
        rule = self._rules.get(self._tenant)
        if isinstance(rule, dict):  # column allow-list per table
            cols = rule.get(table)
            return cols is None or column in cols
        if isinstance(rule, list):  # table allow-list
            return table in rule
        return True

    def filter_column_items(self, items, name_attr: str = "name",
                            table_attr: Optional[str] = "table") -> List:
        """Filter a list of column-like objects by tenant permission."""
        if not self.enabled:
            return list(items)
        out = []
        for it in items:
            table = getattr(it, table_attr, "") if table_attr else ""
            col = getattr(it, name_attr, "")
            if self.can_access_column(table, col):
                out.append(it)
        return out
