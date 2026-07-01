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

"""Tests for incremental_merge — LightRAG-style merge utilities."""

import pytest

from hugegraph_llm.operators.graph_op.incremental_merge import (
    EntityMergeInput,
    EntityMergeOutput,
    IncrementalMergePipeline,
    RelationMergeInput,
    RelationMergeOutput,
    SourceIdStrategy,
    SourceIdsConfig,
    SourceIdsManager,
    merge_entity_descriptions,
    vote_entity_type,
)
from hugegraph_llm.operators.llm_op.description_merger import (
    DescriptionMerger,
    DescriptionMergerConfig,
)


# ---------------------------------------------------------------------------
# SourceIdsManager
# ---------------------------------------------------------------------------


class TestSourceIdsManager:
    """Source ID deduplication and limit strategies."""

    def test_merge_new_ids_into_empty(self):
        mgr = SourceIdsManager()
        result = mgr.merge(existing_ids=[], new_ids=["doc1", "doc2"])
        assert result == ["doc1", "doc2"]

    def test_merge_dedup_preserves_order(self):
        mgr = SourceIdsManager()
        result = mgr.merge(existing_ids=["doc1", "doc2"], new_ids=["doc2", "doc3"])
        assert result == ["doc1", "doc2", "doc3"]

    def test_merge_none_inputs(self):
        mgr = SourceIdsManager()
        result = mgr.merge(existing_ids=None, new_ids=None)
        assert result == []

    def test_keep_strategy_truncates_from_start(self):
        mgr = SourceIdsManager(config=SourceIdsConfig(max_source_ids=3, strategy=SourceIdStrategy.KEEP))
        result = mgr.merge(
            existing_ids=["doc1", "doc2", "doc3"],
            new_ids=["doc4", "doc5"],
        )
        assert result == ["doc1", "doc2", "doc3"]

    def test_fifo_strategy_truncates_from_end(self):
        mgr = SourceIdsManager(config=SourceIdsConfig(max_source_ids=3, strategy=SourceIdStrategy.FIFO))
        result = mgr.merge(
            existing_ids=["doc1", "doc2", "doc3"],
            new_ids=["doc4", "doc5"],
        )
        assert result == ["doc3", "doc4", "doc5"]

    def test_no_limit_needed(self):
        mgr = SourceIdsManager(config=SourceIdsConfig(max_source_ids=200))
        result = mgr.merge(existing_ids=["doc1"], new_ids=["doc2"])
        assert len(result) == 2  # well below limit

    def test_remove_specific_ids(self):
        mgr = SourceIdsManager()
        result = mgr.remove(["doc1", "doc2", "doc3"], ["doc2"])
        assert result == ["doc1", "doc3"]

    def test_default_config(self):
        config = SourceIdsConfig()
        assert config.max_source_ids == 200
        assert config.strategy == SourceIdStrategy.KEEP


# ---------------------------------------------------------------------------
# vote_entity_type
# ---------------------------------------------------------------------------


class TestVoteEntityType:
    """Majority-vote entity type resolution."""

    def test_existing_type_only(self):
        result = vote_entity_type(existing_type="Person")
        assert result == "Person"

    def test_new_type_only(self):
        result = vote_entity_type(existing_type=None, new_types=["Organization"])
        assert result == "Organization"

    def test_majority_vote(self):
        result = vote_entity_type(existing_type="Person", new_types=["Person", "Organization"])
        assert result == "Person"  # Person appears 2x, Organization 1x

    def test_all_types_override(self):
        result = vote_entity_type(all_types=["Org", "Org", "Person", "Org"])
        assert result == "Org"  # Org appears 3x

    def test_no_types(self):
        result = vote_entity_type(existing_type=None, new_types=None, all_types=None)
        assert result == ""

    def test_empty_types_filtered(self):
        result = vote_entity_type(all_types=["", "Person", "", "Person"])
        assert result == "Person"

    def test_tie_returns_first(self):
        """When tied, Counter.most_common returns the first encountered."""
        result = vote_entity_type(all_types=["A", "B"])
        assert result in ["A", "B"]


# ---------------------------------------------------------------------------
# merge_entity_descriptions
# ---------------------------------------------------------------------------


class TestMergeEntityDescriptions:
    """Description merge using DescriptionMerger four-level strategy."""

    def test_single_existing_description(self):
        result = merge_entity_descriptions(existing_description="Alice is a researcher.")
        # merge() returns a string; single desc → returned directly
        assert result == "Alice is a researcher."

    def test_existing_plus_new(self):
        result = merge_entity_descriptions(
            existing_description="Alice works at MIT.",
            new_descriptions=["Alice published 3 papers."],
        )
        # Two short descriptions → join with separator
        assert isinstance(result, str)
        assert "Alice works at MIT." in result

    def test_no_existing(self):
        result = merge_entity_descriptions(new_descriptions=["New description."])
        assert result == "New description."

    def test_empty_list(self):
        result = merge_entity_descriptions(existing_description=None, new_descriptions=None)
        # DescriptionMerger with empty list → returns ""
        assert isinstance(result, str)

    def test_custom_merger(self):
        merger = DescriptionMerger(config=DescriptionMergerConfig(separator=" || "))
        result = merge_entity_descriptions(
            existing_description="desc1",
            new_descriptions=["desc2"],
            merger=merger,
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# IncrementalMergePipeline — Entity merge
# ---------------------------------------------------------------------------


class TestPipelineEntityMerge:
    """Full entity merge pipeline: type vote + description + source_ids."""

    def test_new_entity(self):
        pipeline = IncrementalMergePipeline()
        input_ = EntityMergeInput(
            entity_name="Alice",
            new_type="Person",
            new_description="Alice is a researcher.",
            new_source_ids=["doc1"],
        )
        output = pipeline.merge_entity(input_)
        assert output.merged_type == "Person"
        assert output.merged_description == "Alice is a researcher."
        assert output.merged_source_ids == ["doc1"]
        assert output.changed is True

    def test_entity_merge_with_existing(self):
        pipeline = IncrementalMergePipeline()
        input_ = EntityMergeInput(
            entity_name="Alice",
            existing_type="Person",
            existing_description="Alice works at MIT.",
            existing_source_ids=["doc1"],
            new_type="Person",
            new_description="Alice published 3 papers.",
            new_source_ids=["doc2"],
        )
        output = pipeline.merge_entity(input_)
        assert output.merged_type == "Person"  # Person wins (2x)
        assert len(output.merged_source_ids) == 2
        assert "doc1" in output.merged_source_ids
        assert "doc2" in output.merged_source_ids
        assert output.changed is True

    def test_entity_no_change(self):
        """When existing and new are identical, changed=False or source_ids unchanged."""
        pipeline = IncrementalMergePipeline()
        input_ = EntityMergeInput(
            entity_name="Alice",
            existing_type="Person",
            existing_description="Alice is a researcher.",
            existing_source_ids=["doc1"],
            new_type="Person",
            new_description="Alice is a researcher.",  # same description
            new_source_ids=["doc1"],  # same source (dedup removes)
        )
        output = pipeline.merge_entity(input_)
        assert output.merged_source_ids == ["doc1"]

    def test_source_ids_limit_applied(self):
        pipeline = IncrementalMergePipeline(
            source_ids_config=SourceIdsConfig(max_source_ids=2, strategy=SourceIdStrategy.KEEP),
        )
        input_ = EntityMergeInput(
            entity_name="Alice",
            existing_source_ids=["doc1", "doc2"],
            new_source_ids=["doc3"],
        )
        output = pipeline.merge_entity(input_)
        assert len(output.merged_source_ids) == 2
        assert output.merged_source_ids == ["doc1", "doc2"]  # KEEP strategy

    def test_type_conflict_resolution(self):
        """Different types: majority vote resolves."""
        pipeline = IncrementalMergePipeline()
        input_ = EntityMergeInput(
            entity_name="MIT",
            existing_type="Organization",
            new_type="University",
        )
        output = pipeline.merge_entity(input_)
        assert output.merged_type in ["Organization", "University"]


# ---------------------------------------------------------------------------
# IncrementalMergePipeline — Relation merge
# ---------------------------------------------------------------------------


class TestPipelineRelationMerge:
    """Full relation merge pipeline: description + source_ids."""

    def test_new_relation(self):
        pipeline = IncrementalMergePipeline()
        input_ = RelationMergeInput(
            source_entity="Alice",
            target_entity="MIT",
            relation_label="works_at",
            new_description="Alice works at MIT as a researcher.",
            new_source_ids=["doc1"],
        )
        output = pipeline.merge_relation(input_)
        assert output.merged_description == "Alice works at MIT as a researcher."
        assert output.merged_source_ids == ["doc1"]
        assert output.changed is True

    def test_relation_merge_with_existing(self):
        pipeline = IncrementalMergePipeline()
        input_ = RelationMergeInput(
            source_entity="Alice",
            target_entity="MIT",
            relation_label="works_at",
            existing_description="Alice is employed at MIT.",
            existing_source_ids=["doc1"],
            new_description="Alice is a professor at MIT.",
            new_source_ids=["doc2"],
        )
        output = pipeline.merge_relation(input_)
        assert len(output.merged_source_ids) == 2
        assert output.changed is True

    def test_relation_no_change(self):
        pipeline = IncrementalMergePipeline()
        input_ = RelationMergeInput(
            source_entity="Alice",
            target_entity="MIT",
            relation_label="works_at",
            existing_description="Same desc",
            existing_source_ids=["doc1"],
            new_description="Same desc",
            new_source_ids=["doc1"],  # duplicate
        )
        output = pipeline.merge_relation(input_)
        assert output.merged_source_ids == ["doc1"]
