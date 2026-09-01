"""DEPRECATED location — the NL2SQL write path now lives in the src package:

    hugegraph-llm/src/hugegraph_llm/nl2sql/ingest_to_hg.py

This file is kept as a thin re-export so existing scripts (merge_step*.py,
e2e_ingest_load.py, ...) keep working unchanged. New code should import from
``hugegraph_llm.nl2sql.ingest_to_hg``, and the production entry point is
``hugegraph_llm.nl2sql.ingest.Nl2SqlIngester`` (single interface for
structured metadata AND documents, single graph, single vector store).

Run (unchanged): PYTHONPATH=incubator-hugegraph-ai/hugegraph-llm/src \
       /path/to/hg-llm/python scripts/ingest_metadata_to_hg.py --meta ...
"""
from hugegraph_llm.nl2sql.ingest_to_hg import (  # noqa: F401
    DATALOG_RULES,
    LOG_PATH,
    _clear_graph,
    _ensure_schema,
    _fetch_ids,
    _log_validation,
    _request,
    bare_table,
    ingest,
    log,
    validate_metadata_rules,
)

__all__ = [
    "DATALOG_RULES", "LOG_PATH", "bare_table", "ingest", "log",
    "validate_metadata_rules",
]
