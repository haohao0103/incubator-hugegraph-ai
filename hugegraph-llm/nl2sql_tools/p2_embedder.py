"""Local, offline sentence-transformers embedder for P2 semantic linking.

Wraps a Chinese-capable ST model as ``Callable[[str], List[float]]`` so it can
be plugged straight into ``SchemaLinker(embedder=...)``.

Usage:
    from p2_embedder import make_embedder
    embedder = make_embedder()                      # model from $P2_MODEL or default
    vec = embedder("支付总额")                       # List[float]

The model is loaded once (lazy singleton) and reused. No network at call time
once downloaded. Point ``HF_ENDPOINT`` at a mirror (e.g. https://hf-mirror.com)
only for the initial download.
"""
import os
import threading
from typing import Callable, List, Optional

_DEFAULT_MODEL = "shibing624/text2vec-base-chinese"

_lock = threading.Lock()
_model = None  # type: ignore


def _load_model(name: str):
    global _model
    with _lock:
        if _model is not None:
            return _model
        # Force offline: the model must already be cached locally. huggingface.co
        # is unreachable on this network, and SentenceTransformer would otherwise
        # stall on freshness HEAD checks (each with 5 retries x 10s timeout).
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        # Import lazily so this module is importable even without torch/ST.
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name)
        return _model


def make_embedder(model_name: Optional[str] = None) -> Callable[[str], List[float]]:
    """Return a ``Callable[[str], List[float]]`` over the given ST model."""
    name = model_name or os.environ.get("P2_MODEL", _DEFAULT_MODEL)
    mdl = _load_model(name)

    def _embed(text: str) -> List[float]:
        vec = mdl.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return vec.tolist()

    return _embed


def model_dimension(model_name: Optional[str] = None) -> int:
    name = model_name or os.environ.get("P2_MODEL", _DEFAULT_MODEL)
    mdl = _load_model(name)
    return mdl.get_sentence_embedding_dimension()
