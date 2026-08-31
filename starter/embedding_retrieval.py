"""Optional local semantic candidate route for hybrid retrieval.

There are two distinct computation phases:

1. Startup/cache miss: encode all 50,000 product documents once and save the
   normalized matrix to disk.
2. Query time: encode only the distilled current intent and compute cosine
   similarity against the cached matrix with one NumPy matrix multiplication.

This route adds candidates for meaning-level matches; lexical retrieval remains
the authority for exact brand, size, color, material, and budget constraints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_VERSION = 1
DOCUMENT_FIELDS = (
    "title",
    "categories",
    "store",
    "features",
    "details",
    "description",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {_text(item)}" for key, item in value.items())
    if isinstance(value, set):
        return " ".join(_text(item) for item in sorted(value, key=str))
    if isinstance(value, (list, tuple)):
        return " ".join(_text(item) for item in value)
    return str(value)


def product_document(product: object) -> str:
    """Build the bounded, field-labelled text encoded for one product."""

    if not isinstance(product, dict):
        return ""
    parts: list[str] = []
    for field in DOCUMENT_FIELDS:
        value = _text(product.get(field)).strip()
        if value:
            parts.append(f"{field}: {value}")
    # MiniLM truncates long inputs, so preserve high-value fields at the front.
    return re.sub(r"\s+", " ", ". ".join(parts)).strip()[:1800]


def semantic_query(
    query: object,
    category: object = None,
    active_constraints: object = None,
) -> str:
    """Distill the current dialog state into a semantic-search query."""

    parts: list[str] = []
    message = re.sub(r"\s+", " ", str(query or "")).strip()
    if message:
        parts.append(message)
    category_text = re.sub(r"\s+", " ", str(category or "")).strip()
    if category_text and category_text.lower() not in message.lower():
        parts.append(f"category: {category_text}")
    if isinstance(active_constraints, dict):
        for attribute, values in sorted(
            active_constraints.items(), key=lambda item: str(item[0])
        ):
            items = values if isinstance(values, (list, tuple, set)) else [values]
            cleaned = [
                re.sub(r"\s+", " ", str(value)).strip()
                for value in items
                if str(value).strip()
            ]
            if cleaned:
                parts.append(f"{attribute}: {'; '.join(cleaned)}")
    return ". ".join(parts)[:1200]


def _catalog_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_slug(model_name: str) -> str:
    name = Path(model_name).name or model_name
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "model"


class EmbeddingRetriever:
    """Optional local dense retriever with a reusable NumPy embedding cache.

    Heavy dependencies are imported lazily so the standard-library BM25 agent
    remains usable when this experimental route is disabled.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        model_name_or_path: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        batch_size: int = 128,
    ) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Embedding retrieval requires sentence-transformers and numpy."
            ) from error

        self._np = np
        self.catalog_path = Path(catalog_path).resolve()
        self.model_name_or_path = str(model_name_or_path)
        self.batch_size = max(1, int(batch_size))
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")

        self.model = SentenceTransformer(
            self.model_name_or_path,
            device="cpu",
            local_files_only=True,
        )
        root = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else self.catalog_path.parent / ".embedding_cache"
        )
        root.mkdir(parents=True, exist_ok=True)
        fingerprint = _catalog_digest(self.catalog_path)[:16]
        cache_name = f"v{CACHE_VERSION}-{_model_slug(self.model_name_or_path)}-{fingerprint}"
        self.cache_path = root / cache_name
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.ids_path = self.cache_path / "parent_asins.json"
        self.embeddings_path = self.cache_path / "embeddings.npy"

        # Product embeddings are expensive but catalog-stable. The digest makes
        # stale caches impossible to reuse with a different frozen catalog.
        if not self._load_cache():
            self._build_cache()
            if not self._load_cache():
                raise RuntimeError("Embedding cache could not be loaded after creation.")

    def _load_cache(self) -> bool:
        if not self.ids_path.is_file() or not self.embeddings_path.is_file():
            return False
        try:
            identifiers = json.loads(self.ids_path.read_text(encoding="utf-8"))
            embeddings = self._np.load(self.embeddings_path, mmap_mode="r")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if (
            not isinstance(identifiers, list)
            or embeddings.ndim != 2
            or len(identifiers) != embeddings.shape[0]
        ):
            return False
        self.parent_asins = tuple(str(value) for value in identifiers)
        self.embeddings = embeddings
        return True

    def _encode_documents(self, documents: list[str]):
        encoder = getattr(self.model, "encode_document", None)
        if encoder is None:
            encoder = self.model.encode
        return encoder(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _build_cache(self) -> None:
        """Encode the catalog in batches and persist its normalized vectors."""

        identifiers: list[str] = []
        chunks: list[object] = []
        documents: list[str] = []

        def flush() -> None:
            if not documents:
                return
            encoded = self._encode_documents(documents)
            chunks.append(self._np.asarray(encoded, dtype=self._np.float32))
            documents.clear()

        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product.get("parent_asin", "")).strip()
                if not parent_asin:
                    continue
                identifiers.append(parent_asin)
                documents.append(product_document(product))
                if len(documents) >= self.batch_size:
                    flush()
        flush()
        if not identifiers:
            raise ValueError("Catalog contains no valid products to embed.")
        matrix = self._np.concatenate(chunks, axis=0).astype(self._np.float32, copy=False)
        self._np.save(self.embeddings_path, matrix, allow_pickle=False)
        self.ids_path.write_text(
            json.dumps(identifiers, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def retrieve(self, query: object, *, top_k: int = 200) -> list[dict]:
        """Return lightweight nearest IDs/scores for later catalog hydration."""

        try:
            limit = max(0, min(int(top_k), len(self.parent_asins)))
        except (TypeError, ValueError):
            return []
        text = re.sub(r"\s+", " ", str(query or "")).strip()
        if not text or limit == 0:
            return []

        encoder = getattr(self.model, "encode_query", None)
        if encoder is None:
            encoder = self.model.encode
        vector = encoder(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vector = self._np.asarray(vector, dtype=self._np.float32).reshape(-1)
        # Both sides are L2-normalized; dot product is therefore cosine
        # similarity. Exact search over 50k x 384 remains small enough that a
        # vector database would add unnecessary infrastructure.
        scores = self.embeddings @ vector
        if limit == len(self.parent_asins):
            indices = self._np.arange(len(self.parent_asins))
        else:
            indices = self._np.argpartition(scores, -limit)[-limit:]
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: (-float(scores[index]), self.parent_asins[index]),
        )
        results: list[dict] = []
        for index in ordered:
            score = float(scores[index])
            if not math.isfinite(score):
                continue
            results.append({
                "parent_asin": self.parent_asins[index],
                "semantic_similarity": max(-1.0, min(1.0, score)),
                "route_hits": ["semantic"],
            })
        return results

    def close(self) -> None:
        # NumPy memory maps are released with the retriever instance.
        self.embeddings = None
