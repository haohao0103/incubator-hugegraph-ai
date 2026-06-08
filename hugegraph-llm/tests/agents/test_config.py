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

"""Tests for graphrag-related config additions."""

import pytest

from hugegraph_llm.config.hugegraph_config import HugeGraphConfig
from hugegraph_llm.config.llm_config import LLMConfig
from hugegraph_llm.config.prompt_config import PromptConfig


# ── HugeGraphConfig Tests ──────────────────────────────────────


class TestHugeGraphConfigNewFields:
    """Tests for the new graphrag fields in HugeGraphConfig.

    Note: The .env file overrides instance values at runtime. These tests
    verify class-level default definitions and the presence of fields.
    """

    def test_community_detection_algorithm_field_exists(self):
        """Test community_detection_algorithm field is defined with default 'leiden'."""
        default = HugeGraphConfig.model_fields["community_detection_algorithm"].default
        assert default == "leiden"

    def test_max_community_levels_field_exists(self):
        """Test max_community_levels field is defined with default 2."""
        default = HugeGraphConfig.model_fields["max_community_levels"].default
        assert default == 2

    def test_min_community_size_field_exists(self):
        """Test min_community_size field is defined with default 3."""
        default = HugeGraphConfig.model_fields["min_community_size"].default
        assert default == 3

    def test_community_resolution_field_exists(self):
        """Test community_resolution field is defined with default 1.0."""
        default = HugeGraphConfig.model_fields["community_resolution"].default
        assert default == 1.0

    def test_max_community_reports_field_exists(self):
        """Test max_community_reports field is defined with default 100."""
        default = HugeGraphConfig.model_fields["max_community_reports"].default
        assert default == 100

    def test_enable_provenance_field_exists(self):
        """Test enable_provenance field is defined with default False."""
        default = HugeGraphConfig.model_fields["enable_provenance"].default
        assert default is False

    def test_config_instantiation(self):
        """Test that config can be instantiated (env file may override)."""
        config = HugeGraphConfig()
        assert isinstance(config.community_detection_algorithm, str)
        assert isinstance(config.max_community_levels, int)
        assert isinstance(config.enable_provenance, bool)

    def test_legacy_fields_still_exist(self):
        """Test that legacy config fields still exist in config model."""
        config = HugeGraphConfig()
        assert config.graph_url is not None
        assert config.graph_name is not None
        assert isinstance(config.max_graph_path, int)

    def test_model_dump_includes_new_fields(self):
        """Test that model_dump() includes new fields."""
        config = HugeGraphConfig()
        d = config.model_dump()
        assert "community_detection_algorithm" in d
        assert "max_community_levels" in d
        assert "min_community_size" in d
        assert "enable_provenance" in d


# ── LLMConfig Tests ────────────────────────────────────────────


class TestLLMConfigAgentFields:
    """Tests for the new agent_llm fields in LLMConfig.

    Note: The .env file overrides instance values at runtime. These tests
    verify class-level default definitions and the presence of fields.
    """

    def test_agent_llm_type_field_exists(self):
        """Test agent_llm_type field is defined with default 'openai'."""
        default = LLMConfig.model_fields["agent_llm_type"].default
        assert default == "openai"

    def test_openai_agent_api_base_field_exists(self):
        """Test openai_agent_api_base field is defined."""
        field = LLMConfig.model_fields["openai_agent_api_base"]
        assert field.default is not None
        assert "api.openai.com" in str(field.default) or "openai" in str(field.default_factory)

    def test_openai_agent_language_model_default(self):
        """Test openai_agent_language_model default."""
        default = LLMConfig.model_fields["openai_agent_language_model"].default
        assert default == "gpt-4.1-mini"

    def test_openai_agent_tokens_default(self):
        """Test openai_agent_tokens default."""
        default = LLMConfig.model_fields["openai_agent_tokens"].default
        assert default == 8192

    def test_ollama_agent_fields_exist(self):
        """Test that Ollama agent fields are present."""
        config = LLMConfig()
        assert hasattr(config, "ollama_agent_host")
        assert hasattr(config, "ollama_agent_port")
        assert hasattr(config, "ollama_agent_language_model")

    def test_litellm_agent_fields_exist(self):
        """Test that LiteLLM agent fields are present with correct defaults."""
        config = LLMConfig()
        assert hasattr(config, "litellm_agent_api_key")
        assert hasattr(config, "litellm_agent_api_base")
        assert hasattr(config, "litellm_agent_language_model")
        assert hasattr(config, "litellm_agent_tokens")

    def test_openai_agent_language_model_default(self):
        """Test openai_agent_language_model default."""
        config = LLMConfig()
        assert config.openai_agent_language_model == "gpt-4.1-mini"

    def test_openai_agent_tokens_default(self):
        """Test openai_agent_tokens default."""
        config = LLMConfig()
        assert config.openai_agent_tokens == 8192

    def test_ollama_agent_fields_exist(self):
        """Test that Ollama agent fields are present."""
        config = LLMConfig()
        assert config.ollama_agent_host == "127.0.0.1"
        assert config.ollama_agent_port == 11434
        assert config.ollama_agent_language_model is None

    def test_litellm_agent_fields_exist(self):
        """Test that LiteLLM agent fields are present."""
        config = LLMConfig()
        assert config.litellm_agent_api_key is None
        assert config.litellm_agent_api_base is None
        assert config.litellm_agent_language_model == "openai/gpt-4.1-mini"
        assert config.litellm_agent_tokens == 8192

    def test_legacy_llm_type_fields_exist(self):
        """Test that legacy LLM type fields are defined."""
        config = LLMConfig()
        assert hasattr(config, "chat_llm_type")
        assert hasattr(config, "extract_llm_type")
        assert hasattr(config, "text2gql_llm_type")


# ── PromptConfig Tests ────────────────────────────────────────


class TestPromptConfigNewPrompts:
    """Tests for the new graphrag prompts in PromptConfig."""

    def _make_config(self):
        """Create a PromptConfig with a mock LLM config."""
        mock_llm_config = type("MockLLMConfig", (), {"language": "EN"})()
        return PromptConfig(mock_llm_config)

    def test_agent_system_prompt_en_exists(self):
        """Test that agent_system_prompt_EN is defined."""
        config = self._make_config()
        prompt = config.agent_system_prompt_EN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "knowledge graph analysis agent" in prompt.lower()
        assert "{tool_descriptions}" in prompt
        assert "{conversation_history}" in prompt
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Final Answer:" in prompt

    def test_agent_system_prompt_cn_exists(self):
        """Test that agent_system_prompt_CN is defined."""
        config = self._make_config()
        prompt = config.agent_system_prompt_CN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "知识图谱" in prompt
        assert "{tool_descriptions}" in prompt
        assert "{conversation_history}" in prompt

    def test_query_classifier_prompt_en_exists(self):
        """Test that query_classifier_prompt_EN is defined."""
        config = self._make_config()
        prompt = config.query_classifier_prompt_EN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "simple" in prompt
        assert "complex" in prompt
        assert "{query}" in prompt

    def test_query_classifier_prompt_cn_exists(self):
        """Test that query_classifier_prompt_CN is defined."""
        config = self._make_config()
        prompt = config.query_classifier_prompt_CN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "简单" in prompt
        assert "复杂" in prompt
        assert "{query}" in prompt

    def test_community_report_prompt_en_exists(self):
        """Test that community_report_prompt_EN is defined."""
        config = self._make_config()
        prompt = config.community_report_prompt_EN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "{community_size}" in prompt
        assert "{community_density}" in prompt
        assert "{entity_list}" in prompt
        assert "{relationship_list}" in prompt

    def test_community_report_prompt_cn_exists(self):
        """Test that community_report_prompt_CN is defined."""
        config = self._make_config()
        prompt = config.community_report_prompt_CN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "社区" in prompt
        assert "{community_size}" in prompt

    def test_global_search_map_prompt_en_exists(self):
        """Test that global_search_map_prompt_EN is defined."""
        config = self._make_config()
        prompt = config.global_search_map_prompt_EN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "{query}" in prompt
        assert "{community_title}" in prompt
        assert "{report_text}" in prompt

    def test_global_search_map_prompt_cn_exists(self):
        """Test that global_search_map_prompt_CN is defined."""
        config = self._make_config()
        prompt = config.global_search_map_prompt_CN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "{query}" in prompt

    def test_global_search_reduce_prompt_en_exists(self):
        """Test that global_search_reduce_prompt_EN is defined."""
        config = self._make_config()
        prompt = config.global_search_reduce_prompt_EN
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "{query}" in prompt
        assert "{findings_text}" in prompt

    def test_legacy_prompts_still_exist(self):
        """Test that legacy prompts are unchanged."""
        config = self._make_config()
        assert "answer_prompt_EN" in config.__class__.__dict__
        assert "extract_graph_prompt_EN" in config.__class__.__dict__
        assert "keywords_extract_prompt_EN" in config.__class__.__dict__


# ── WkFlowState / WkFlowInput Tests ────────────────────────────


class TestWkFlowStateNewFields:
    """Tests for the new graphrag fields in WkFlowState."""

    def test_agent_state_fields_exist(self):
        """Test that agent state fields are present on WkFlowState."""
        from hugegraph_llm.state.ai_state import WkFlowState
        state = WkFlowState()
        assert hasattr(state, "agent_answer")
        assert hasattr(state, "agent_trace")
        assert hasattr(state, "agent_total_steps")
        assert hasattr(state, "agent_is_simple_query")
        assert hasattr(state, "agent_error")

    def test_community_state_fields_exist(self):
        """Test that community state fields exist."""
        from hugegraph_llm.state.ai_state import WkFlowState
        state = WkFlowState()
        assert hasattr(state, "communities")
        assert hasattr(state, "community_count")
        assert hasattr(state, "community_reports")
        assert hasattr(state, "community_matches")
        assert hasattr(state, "community_index_built")
        assert hasattr(state, "community_index_count")
        assert hasattr(state, "global_answer")
        assert hasattr(state, "map_findings")
        assert hasattr(state, "communities_used")

    def test_provenance_state_fields_exist(self):
        """Test that provenance state fields exist."""
        from hugegraph_llm.state.ai_state import WkFlowState
        state = WkFlowState()
        assert hasattr(state, "include_provenance")
        assert hasattr(state, "provenance_records")
        assert hasattr(state, "citations")
        assert hasattr(state, "provenance_link_count")
        assert hasattr(state, "doc_id")

    def test_setup_resets_new_fields(self):
        """Test that setup() resets all new fields to None."""
        from hugegraph_llm.state.ai_state import WkFlowState
        state = WkFlowState()

        # Set some values
        state.agent_answer = "answer"
        state.communities = [{"id": "C0"}]
        state.provenance_records = [{"doc": "test"}]

        # Reset
        state.setup()

        assert state.agent_answer is None
        assert state.communities is None
        assert state.provenance_records is None

    def test_to_json_skips_none_fields(self):
        """Test that to_json skips None fields."""
        from hugegraph_llm.state.ai_state import WkFlowState
        state = WkFlowState()
        state.agent_answer = "test answer"
        state.community_count = 5

        j = state.to_json()
        assert j["agent_answer"] == "test answer"
        assert j["community_count"] == 5
        assert "agent_trace" not in j  # None, should be skipped

    def test_assign_from_json_sets_new_fields(self):
        """Test that assign_from_json sets new fields correctly."""
        from hugegraph_llm.state.ai_state import WkFlowState
        state = WkFlowState()

        data = {
            "agent_answer": "Response",
            "community_count": 3,
            "provenance_link_count": 10,
        }
        state.assign_from_json(data)

        assert state.agent_answer == "Response"
        assert state.community_count == 3
        assert state.provenance_link_count == 10

    def test_wkflow_input_agent_fields(self):
        """Test that WkFlowInput has agent fields."""
        from hugegraph_llm.state.ai_state import WkFlowInput
        inp = WkFlowInput()
        assert hasattr(inp, "max_steps")
        assert hasattr(inp, "tools_filter")
        assert inp.max_steps is None
        assert inp.tools_filter is None
