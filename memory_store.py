"""
memory_store.py — Procedural memory (style + exemplar) for the generator.

See docs/procedural_memory_design.md. Two subsystems, both backed by an
in-process index that is primed from Postgres at run start and persisted back
via the normal question insert:

  * Style memory      — overused openers per (profession, region), plus a
                        near-duplicate check against recent same-cluster
                        embeddings.
  * Exemplar memory   — a bank of high-quality past questions retrieved (from a
                        similar persona but a DIFFERENT topic) as few-shot
                        examples of voice/register.

Threading model: `embed()` is called from worker threads (via asyncio.to_thread)
and is guarded by a lock; every mutation (`record`) and every read (`get_context`,
`is_near_dup`) happens on the event-loop thread with no awaits, so the shared
index is never touched by two coroutines at once — no extra locking needed there.
"""

import logging
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np

import db

logger = logging.getLogger(__name__)


@dataclass
class _Record:
    vec: np.ndarray | None   # float32[dim], L2-normalized, or None if embed failed
    profession: str
    region: str
    topic: str
    text: str
    score: float
    opener: str


def score_from_flags(flags: dict) -> float:
    """
    Scalar question quality used for the exemplar bank.

        score = (judge PASS ? 1 : 0) + bengali_ratio - (duplicate ? 0.5 : 0)
    """
    flags = flags or {}
    judge = (flags.get("judge") or {})
    passed = 1.0 if str(judge.get("verdict", "")).upper() == "PASS" else 0.0
    bengali = float(flags.get("bengali_ratio", 0.0) or 0.0)
    dup_pen = 0.5 if flags.get("duplicate") else 0.0
    return passed + bengali - dup_pen


class MemoryStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self._model = None
        self._lock = threading.Lock()

        self.records: list[_Record] = []
        self.by_cluster: dict[tuple, list[int]] = {}   # (prof,region) -> record idxs
        self.opener_counts: dict[tuple, Counter] = {}  # (prof,region) -> Counter
        self.exemplar_idx: list[int] = []              # idxs with score >= threshold

    # ─── Embedder ────────────────────────────────────────────────────────────

    def _ensure_model(self):
        """Lazy-load the embedder; degrade to disabled on any failure."""
        if self._model is not None or not self.enabled:
            return
        try:
            from sentence_transformers import SentenceTransformer
            device = self.cfg.get("device", "cuda")
            try:
                self._model = SentenceTransformer(self.cfg["model_name"], device=device)
                logger.info("Memory embedder loaded on %s: %s", device, self.cfg["model_name"])
            except Exception:
                logger.warning("Embedder failed on %s; retrying on cpu.", device)
                self._model = SentenceTransformer(self.cfg["model_name"], device="cpu")
                logger.info("Memory embedder loaded on cpu.")
        except Exception:
            logger.exception("Could not load embedder; disabling procedural memory.")
            self.enabled = False

    def embed(self, text: str) -> np.ndarray | None:
        """Return an L2-normalized float32 embedding, or None if unavailable."""
        if not self.enabled:
            return None
        self._ensure_model()
        if self._model is None:
            return None
        try:
            with self._lock:
                v = self._model.encode([text], normalize_embeddings=True,
                                       show_progress_bar=False)[0]
            return np.asarray(v, dtype=np.float32)
        except Exception:
            logger.exception("Embedding failed; returning None.")
            return None

    # ─── Openers ─────────────────────────────────────────────────────────────

    def opener(self, text: str) -> str:
        """First N whitespace tokens, normalized — a coarse phrasing fingerprint."""
        norm = unicodedata.normalize("NFKC", text).strip().lower()
        return " ".join(norm.split()[: self.cfg.get("opener_tokens", 6)])

    # ─── Priming from DB ─────────────────────────────────────────────────────

    def prime(self, conn):
        """Load prior questions' embeddings/openers/scores into the index."""
        if not self.enabled:
            return
        rows = db.fetch_memory_rows(conn)
        for r in rows:
            vec = None
            if r.get("embedding") is not None:
                vec = np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
            rec = _Record(
                vec=vec,
                profession=r.get("profession") or "",
                region=r.get("region") or "",
                topic=r.get("topic") or "",
                text=r.get("question_text") or "",
                score=float(r.get("quality_score") or 0.0),
                opener=r.get("opener") or self.opener(r.get("question_text") or ""),
            )
            self._index(rec)
        logger.info("Primed procedural memory with %d records (%d exemplars).",
                    len(self.records), len(self.exemplar_idx))

    def _index(self, rec: _Record):
        idx = len(self.records)
        self.records.append(rec)
        key = (rec.profession, rec.region)
        self.by_cluster.setdefault(key, []).append(idx)
        if rec.opener:
            self.opener_counts.setdefault(key, Counter())[rec.opener] += 1
        if rec.score >= self.cfg.get("min_exemplar_score", 1.4):
            self.exemplar_idx.append(idx)

    # ─── Reads (event-loop thread, no awaits) ────────────────────────────────

    def get_context(self, persona_meta: dict) -> dict:
        """
        Return memory context for a persona:
          avoid_openers — openers already repeated in this (profession, region)
          exemplars     — few-shot texts from a similar persona, different topic
        """
        if not self.enabled:
            return {"avoid_openers": [], "exemplars": []}

        prof = persona_meta.get("profession", "")
        region = persona_meta.get("location", "")
        topic = persona_meta.get("pain_point", "")
        key = (prof, region)

        # Overused openers (only warn about ones actually repeated).
        avoid = [op for op, n in self.opener_counts.get(key, Counter()).most_common(
                    self.cfg.get("avoid_openers_n", 5)) if n >= 2]

        exemplars = self._pick_exemplars(prof, region, topic)
        return {"avoid_openers": avoid, "exemplars": exemplars}

    def _pick_exemplars(self, prof: str, region: str, topic: str) -> list[str]:
        k = self.cfg.get("k_exemplars", 3)
        if k <= 0 or not self.exemplar_idx:
            return []

        def candidates(pred):
            recs = [self.records[i] for i in self.exemplar_idx]
            recs = [r for r in recs if r.topic != topic and pred(r)]
            recs.sort(key=lambda r: r.score, reverse=True)
            return recs

        # Prefer same profession, then same region, then anything — always a
        # DIFFERENT topic so we teach voice, not content.
        chosen = candidates(lambda r: r.profession == prof)
        if len(chosen) < k:
            seen = {id(r) for r in chosen}
            chosen += [r for r in candidates(lambda r: r.region == region)
                       if id(r) not in seen]
        if len(chosen) < k:
            seen = {id(r) for r in chosen}
            chosen += [r for r in candidates(lambda r: True) if id(r) not in seen]
        return [r.text for r in chosen[:k]]

    def is_near_dup(self, vec: np.ndarray | None, persona_meta: dict) -> bool:
        """True if `vec` is within sim_threshold of a recent same-cluster vector."""
        if not self.enabled or vec is None:
            return False
        key = (persona_meta.get("profession", ""), persona_meta.get("location", ""))
        idxs = self.by_cluster.get(key, [])
        if not idxs:
            return False
        thr = self.cfg.get("sim_threshold", 0.92)
        # Compare against the most recent few hundred in this cluster.
        for i in idxs[-500:]:
            other = self.records[i].vec
            if other is not None and float(np.dot(vec, other)) >= thr:
                return True
        return False

    # ─── Write (single writer task) ──────────────────────────────────────────

    def record(self, persona_meta: dict, text: str, vec: np.ndarray | None,
               score: float, opener: str = None):
        """Add a freshly-generated question to the in-RAM index."""
        if not self.enabled:
            return
        self._index(_Record(
            vec=vec,
            profession=persona_meta.get("profession", ""),
            region=persona_meta.get("location", ""),
            topic=persona_meta.get("pain_point", ""),
            text=text,
            score=score,
            opener=opener if opener is not None else self.opener(text),
        ))
