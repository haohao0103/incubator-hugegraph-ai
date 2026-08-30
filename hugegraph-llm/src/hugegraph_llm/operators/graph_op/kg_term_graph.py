"""Independent term / synonym layer (术语实体 + 同义词层).

Borrowed from the 货拉拉 metadata-GraphRAG design: business terms are stored
as their own entities with a synonym layer connecting the many ways users
phrase the same concept ("客单价" / "平均每单成交金额" / "avg_order_value").
Keeping this layer *separate* from the metadata KG means:

- the synonym dictionary can be curated/versioned independently of the graph
  schema (no Term vertex labels polluting the metadata KG);
- it can be persisted to JSON and shared between environments;
- it is consumable by every retrieval stage: query understanding
  (:class:`~.kg_query_understanding.QueryUnderstanding`), the multi-recall
  linker (KgSchemaLinker.synonyms), and future entity-level recall.

This module is the structural upgrade of :class:`KgJargonMap`: instead of a
flat slang->canonical dict it keeps *terms* (canonical identifiers) each with
a list of aliases, supports bi-directional lookup, substring matching against
a raw question, JSON persistence, and still converts back to the flat
jargon-map shape that KgSchemaLinker already consumes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hugegraph_llm.operators.graph_op.kg_jargon_map import DEFAULT_JARGON


@dataclass
class TermNode:
    """One business term: the canonical identifier + its aliases."""

    canonical: str
    aliases: List[str] = field(default_factory=list)
    domain: str = ""
    importance: float = 1.0


class KgTermGraph:
    """Canonical term graph with alias edges, independent of the metadata KG.

    Usage::

        terms = KgTermGraph.from_jargon_map({"客单价": "arpu", "单量": "order_count"})
        terms.lookup("客单价")            # -> "arpu"
        terms.match("客单价是多少")        # -> [("客单价", "arpu")]
        terms.expand_terms(["客单", "单价"])  # appends "arpu"
        terms.to_jargon_map()            # flat alias->canonical for KgSchemaLinker

    Persistence::

        terms.save("/path/terms.json")
        loaded = KgTermGraph.load("/path/terms.json")
    """

    def __init__(self, terms: Optional[Iterable[TermNode]] = None) -> None:
        self._canon_to_aliases: Dict[str, List[str]] = {}
        self._alias_to_canon: Dict[str, str] = {}
        self._meta: Dict[str, Any] = {}
        if terms:
            for term in terms:
                self.add_term(term)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_jargon_map(cls, slang_map: Dict[str, str]) -> "KgTermGraph":
        """Build from the flat slang->canonical shape (KgJargonMap compatible)."""
        terms: Dict[str, List[str]] = {}
        for alias, canon in (slang_map or {}).items():
            if not alias or not canon:
                continue
            terms.setdefault(canon, [])
            if alias != canon and alias not in terms[canon]:
                terms[canon].append(alias)
        graph = cls()
        for canon, aliases in terms.items():
            graph.add_term(TermNode(canonical=canon, aliases=aliases))
        return graph

    @classmethod
    def default(cls) -> "KgTermGraph":
        """The curated Huolala domain vocabulary (same source as KgJargonMap)."""
        return cls.from_jargon_map(DEFAULT_JARGON)

    @classmethod
    def load(cls, path: str) -> "KgTermGraph":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        graph = cls.from_jargon_map(payload.get("aliases", {}))
        graph._meta = payload.get("meta", {})
        return graph

    def add_term(self, term: TermNode) -> None:
        canon = str(term.canonical or "").strip()
        if not canon:
            return
        existing = self._canon_to_aliases.setdefault(canon, [])
        for alias in term.aliases:
            alias = str(alias or "").strip()
            if not alias or alias == canon:
                continue
            if alias not in existing:
                existing.append(alias)
            self._alias_to_canon[alias] = canon

    def add_alias(self, canonical: str, alias: str) -> None:
        canon = str(canonical or "").strip()
        alias = str(alias or "").strip()
        if not canon or not alias or alias == canon:
            return
        aliases = self._canon_to_aliases.setdefault(canon, [])
        if alias not in aliases:
            aliases.append(alias)
        self._alias_to_canon[alias] = canon

    # -- queries -------------------------------------------------------------

    def lookup(self, alias: str) -> Optional[str]:
        """Exact alias -> canonical identifier, or None."""
        if not alias:
            return None
        canon = self._alias_to_canon.get(alias)
        if canon:
            return canon
        # a canonical name asked directly resolves to itself
        return alias if alias in self._canon_to_aliases else None

    def canonical_aliases(self, canonical: str) -> List[str]:
        """All aliases of a canonical term (longest first for matching)."""
        return sorted(self._canon_to_aliases.get(canonical, []), key=len, reverse=True)

    def match(self, text: str) -> List[Tuple[str, str]]:
        """Every (alias, canonical) hit found as a substring of ``text``.

        Longer aliases are tested first so "完单量" wins over "完单".
        """
        if not text:
            return []
        hits: List[Tuple[str, str]] = []
        seen_canon: set = set()
        for alias in sorted(self._alias_to_canon, key=len, reverse=True):
            canon = self._alias_to_canon[alias]
            if canon in seen_canon:
                continue  # first (longest) alias for this canonical already hit
            if alias in text:
                hits.append((alias, canon))
                seen_canon.add(canon)
        return hits

    def expand_terms(self, terms: Sequence[str]) -> List[str]:
        """Append the canonical form of any alias in ``terms`` (dedup, order-kept)."""
        out: List[str] = [t for t in terms if t]
        lower = {t.lower() for t in out}
        for term in terms:
            canon = self.lookup(term)
            if canon and canon.lower() not in lower:
                out.append(canon)
                lower.add(canon.lower())
        return out

    def expand_question(self, question: str) -> List[str]:
        """Aliases matched against the raw question -> their canonical names."""
        return [canon for _alias, canon in self.match(question)]

    def to_jargon_map(self) -> Dict[str, str]:
        """Flat alias->canonical shape (KgSchemaLinker.synonyms compatible)."""
        return dict(self._alias_to_canon)

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self._meta,
            "aliases": self.to_jargon_map(),
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    # -- misc ----------------------------------------------------------------

    @property
    def num_terms(self) -> int:
        return len(self._canon_to_aliases)

    @property
    def num_aliases(self) -> int:
        return len(self._alias_to_canon)

    def __len__(self) -> int:
        return self.num_aliases
