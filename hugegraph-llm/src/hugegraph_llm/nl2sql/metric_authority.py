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

"""Metric authority resolution: pick the canonical definition when schema
linking returns several candidate metrics, or when one metric name carries
conflicting definitions across domains.

Borrowed from the parallel NL2SQL demo branch's ``kg_metric_authority``.
Authority is read from a Term/Node's ``properties``:

* ``authoritative`` (bool)  — governance stamp; dominates every other signal;
* ``priority`` (int)        — numeric tie-breaker among authoritative / none;
* ``source`` (str)          — originating system, ranked best -> worst.

Higher ``authority_score`` wins. Pure function, no external deps.
"""

from typing import Any, Dict, List, Optional, Tuple

# Source systems ranked best -> worst. A better source earns a higher bonus.
SOURCE_RANK: Dict[str, int] = {
    "governance": 5,
    "data_platform": 4,
    "metric_mart": 3,
    "business": 2,
    "query_log": 1,
}
DEFAULT_SOURCE_BONUS = 1  # missing / unknown source still allowed, but low
AUTHORITATIVE_BONUS = 1000  # dominates every priority/source combination


def _is_true(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def authority_score(metric_props: Dict[str, Any]) -> Tuple[int, bool]:
    """Return ``(score, is_authoritative)`` for a metric/Term node's properties.

    ``authoritative`` dominates; then ``priority``; then the source bonus.
    """
    is_auth = _is_true(metric_props.get("authoritative"))
    priority = _as_int(metric_props.get("priority", 0))
    source = str(metric_props.get("source", "") or "")
    source_bonus = SOURCE_RANK.get(source, DEFAULT_SOURCE_BONUS)
    score = priority + source_bonus
    if is_auth:
        score += AUTHORITATIVE_BONUS
    return score, is_auth


def resolve_metric(
    candidates: List[Tuple[str, Dict[str, Any]]],
) -> Optional[Tuple[str, int, bool]]:
    """Pick the authoritative metric among ``(name, properties)`` candidates.

    Returns ``(name, score, is_authoritative)`` for the winner, or ``None``
    when there are no candidates. Deterministic: score desc, name asc.
    """
    if not candidates:
        return None
    scored = [(authority_score(props)[0], name, props) for name, props in candidates]
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_name, best_props = scored[0]
    return best_name, best_score, _is_true(best_props.get("authoritative"))
