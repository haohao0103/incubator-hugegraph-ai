"""Cross-encoder reranking for schema linking (two-stage retrieval).

Stage 1 — recall: PPR over the schema graph returns a candidate pool
(``candidate_k`` items, deliberately larger than the final ``top_k``).
Stage 2 — precision: a cross-encoder jointly encodes (question, schema
element) pairs and reorders the pool, then the top-k is returned.

Why a cross-encoder on top of an embedding stage: a bi-encoder compresses
query and document independently, so it cannot model their interaction. A
cross-encoder reads both together and assigns a relevance score directly,
which mainly improves *rank position* — hence MRR benefits the most, and R@k
can improve too because the candidate pool is larger than top_k.

Design notes:
- Synchronous on purpose: the NL2SQL pipeline is synchronous, unlike the
  async reranker under ``operators/graph_op/reranker.py``.
- The model is loaded lazily and the whole stage degrades to a no-op when
  ``sentence-transformers`` or the model is unavailable, so enabling it by
  configuration can never break a running service.
- PPR scores are preserved (``item.score``); the cross-encoder score is
  carried separately in ``item.rerank_score`` so downstream ``min_score``
  gating keeps its original calibration.
"""

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Chinese/English cross-encoder. The sentence-transformers default
# (cross-encoder/ms-marco-MiniLM-L-6-v2) is English-only and performs poorly
# on Chinese warehouse schemas, so it is not used here.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


def item_text(item: Any) -> str:
    """Flatten a :class:`LinkedItem` into the text a cross-encoder scores."""
    props: Dict[str, Any] = getattr(item, "properties", {}) or {}
    table = getattr(item, "table", "") or ""
    comment = props.get("comment", "") or ""
    data_type = props.get("data_type", "") or ""
    name = getattr(item, "name", "") or ""

    if getattr(item, "node_type", "") == "column":
        head = f"{table}.{name}" if table else name
    else:
        head = name
    parts = [p for p in (head, data_type, comment) if p]
    return " ".join(str(p) for p in parts)


class CrossEncoderReranker:
    """Rescore schema-linking candidates with a cross-encoder.

    Args:
        model_name: cross-encoder model id. Defaults to
            :data:`DEFAULT_RERANK_MODEL`.
        candidate_k: how many PPR results to rescore before truncating to the
            requested ``top_k``. Larger = better recall ceiling, slower.
        alpha: when set (0..1), blend with the retrieval score as
            ``alpha * norm(ppr) + (1 - alpha) * norm(ce)``. ``None`` (default)
            means rank purely by the cross-encoder, which is what the
            two-stage setup is for.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        candidate_k: int = 20,
        alpha: Optional[float] = None,
    ) -> None:
        self._model_name = model_name or DEFAULT_RERANK_MODEL
        self._candidate_k = max(candidate_k, 1)
        self._alpha = alpha
        self._model: Optional[object] = None
        self._available: Optional[bool] = None

    # -- introspection ---------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available(self) -> bool:
        if self._available is None:
            self._try_load()
        return bool(self._available)

    def _try_load(self) -> bool:
        """Load the cross-encoder lazily; remember failures permanently."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self._model_name)
            self._available = True
            log.info("nl2sql reranker loaded: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001 -- optional dependency
            self._available = False
            log.warning(
                "nl2sql reranker unavailable (%s); falling back to retrieval order",
                exc,
            )
        return bool(self._available)

    # -- reranking -------------------------------------------------------
    def rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """Rescore ``items`` against ``query`` and return the top ``top_k``.

        Falls back to the incoming order when the model is unavailable or the
        candidate list is empty. Never raises.
        """
        if not items:
            return []
        if self._available is None:
            self._try_load()
        if not self._available or self._model is None:
            return items[:top_k]

        candidates = items[: self._candidate_k]
        pairs = [(query, item_text(it)) for it in candidates]
        try:
            raw = self._model.predict(pairs)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 -- never break retrieval
            log.warning("nl2sql reranker predict failed (%s); keeping order", exc)
            return items[:top_k]

        scores = [float(s) for s in raw]
        return self._merge(candidates, scores, top_k)

    def _merge(self, candidates: List[Any], scores: List[float], top_k: int) -> List[Any]:
        """Attach scores and produce the final ordering."""
        for it, sc in zip(candidates, scores):
            try:
                it.rerank_score = sc
            except Exception:  # noqa: BLE001 -- read-only dataclass
                pass

        if self._alpha is None:
            ranked = sorted(
                zip(candidates, scores), key=lambda p: p[1], reverse=True
            )
            return [it for it, _ in ranked[:top_k]]

        # Blend: normalise both signals to [0, 1] before mixing, otherwise the
        # cross-encoder (unbounded, often negative) would dominate by scale.
        ppr = [float(getattr(it, "score", 0.0) or 0.0) for it in candidates]
        blended = [
            self._alpha * _norm(v, ppr) + (1.0 - self._alpha) * _norm(s, scores)
            for v, s in zip(ppr, scores)
        ]
        ranked = sorted(zip(candidates, blended), key=lambda p: p[1], reverse=True)
        return [it for it, _ in ranked[:top_k]]


def _norm(value: float, population: List[float]) -> float:
    """Min-max normalise ``value`` within ``population`` (0 when flat)."""
    if not population:
        return 0.0
    lo, hi = min(population), max(population)
    if hi - lo < 1e-12:
        return 0.0
    return (value - lo) / (hi - lo)


def make_reranker(
    model_name: Optional[str] = None,
    candidate_k: int = 20,
    alpha: Optional[float] = None,
) -> Optional[CrossEncoderReranker]:
    """Build a reranker, or ``None`` when the dependency/model is missing."""
    reranker = CrossEncoderReranker(
        model_name=model_name, candidate_k=candidate_k, alpha=alpha
    )
    return reranker if reranker.available else None
