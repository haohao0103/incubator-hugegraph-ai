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

"""Typed errors for the skill graph engine.

Every error below is *actionable*: the message says what was inconsistent and
what the operator should do. Nothing here is swallowed silently - the whole
point of these types is to turn previously silent data corruption into a
non-zero exit code.
"""

from __future__ import annotations


class SkillGraphError(Exception):
    """Base class for all skill-graph failures."""


class DuplicateSkillError(SkillGraphError):
    """Two skill files resolve to the same stable id (silent overwrite guard)."""


class EmbeddingMismatchError(SkillGraphError, ValueError):
    """Vectors come from different embedders/dimensions and cannot be compared.

    Subclasses :class:`ValueError` because callers historically guarded against
    dimension problems with ``except ValueError``; keeping that contract lets
    old call sites fail loudly instead of silently changing behaviour.
    """


class GraphWriteError(SkillGraphError):
    """One or more writes to HugeGraph failed (build must not report success)."""


class GraphReadError(SkillGraphError):
    """A stored build could not be read back completely or consistently."""


class BuildValidationError(SkillGraphError):
    """A build artifact or publish request failed its consistency checks."""


class PublishError(SkillGraphError):
    """Activating a build failed; the previous build stays active."""