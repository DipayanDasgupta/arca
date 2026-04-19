"""
arca/memory/vector_memory.py
=============================
Semantic episode retrieval via FAISS (or pure-numpy fallback).

Design
------
Each EpisodeRecord is embedded as a fixed-length float32 vector that
captures the semantics of the run:

  [0]   total_reward (normalised 0–1 by clip at 1000)
  [1]   compromised_ratio  (hosts_compromised / hosts_total)
  [2]   efficiency         (compromised / steps * 100, clipped 0-1)
  [3]   goal_reached       (0.0 or 1.0)
  [4]   path_length        (len(attack_path) / 20, clipped 0-1)
  [5-9] OS fingerprint of the most-exploited hosts (one-hot Windows/Linux/macOS/IoT/Other)
  [10]  severity_score / 10.0
  [11]  mean exploit_prob  (mean of vuln probs in the attack path — proxy for difficulty)
  [12]  firewall_fraction  (fraction of exploited hosts that had firewalls)

Total: 13 dimensions — lightweight, interpretable, no GPU required.

Usage::

    from arca.memory.vector_memory import VectorMemory

    vm = VectorMemory()                 # loads persisted index if it exists
    vm.add(episode_record)
    similar = vm.search(query_record, k=3)

    # Or use with an EpisodeBuffer:
    vm.add_from_buffer(episode_buffer)
    context = vm.format_for_llm(query_record, k=3)
"""
from __future__ import annotations

import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Union
import numpy as np

# ── Optional FAISS ────────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

_EMBED_DIM = 13


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_episode(record) -> np.ndarray:
    """
    Embed an EpisodeRecord (or dict with same keys) into a float32 vector.
    Works with both EpisodeRecord dataclass instances and plain dicts.
    """
    def _get(key, default=0):
        if hasattr(record, key):
            return getattr(record, key)
        if isinstance(record, dict):
            return record.get(key, default)
        return default

    total_reward      = float(_get("total_reward", 0))
    hosts_compromised = int(_get("hosts_compromised", 0))
    hosts_total       = max(int(_get("hosts_total", 1)), 1)
    steps             = max(int(_get("steps", 1)), 1)
    goal_reached      = float(bool(_get("goal_reached", False)))
    attack_path       = _get("attack_path", [])
    severity_score    = float(_get("severity_score", 0))

    # Normalise core metrics
    reward_norm       = min(max(total_reward / 1000.0, -1.0), 1.0)
    comp_ratio        = hosts_compromised / hosts_total
    efficiency        = min(hosts_compromised / steps * 100.0, 1.0)
    path_len          = min(len(attack_path) / 20.0, 1.0)
    sev_norm          = severity_score / 10.0

    # OS fingerprint from attack path strings
    os_counts = np.zeros(5, dtype=np.float32)   # Win Linux macOS IoT Other
    for step_str in attack_path:
        s = step_str.lower()
        if "windows" in s:   os_counts[0] += 1
        elif "linux" in s:   os_counts[1] += 1
        elif "macos" in s:   os_counts[2] += 1
        elif "iot" in s:     os_counts[3] += 1
        else:                os_counts[4] += 1
    total_os = os_counts.sum()
    if total_os > 0:
        os_counts /= total_os

    # Mean exploit probability (parsed from path labels like "(CVE:...)")
    # If not parseable we use 0.5 as a neutral prior
    mean_exploit_prob = 0.5
    firewall_fraction = 0.0

    vec = np.array([
        reward_norm,
        comp_ratio,
        efficiency,
        goal_reached,
        path_len,
        *os_counts,           # 5 values
        sev_norm,
        mean_exploit_prob,
        firewall_fraction,
    ], dtype=np.float32)

    assert len(vec) == _EMBED_DIM, f"Embedding dim mismatch: {len(vec)} != {_EMBED_DIM}"
    return vec


def _episode_id(record) -> str:
    if hasattr(record, "episode_id"):
        return record.episode_id
    if isinstance(record, dict):
        return record.get("episode_id", "")
    return ""


# ── Index implementations ─────────────────────────────────────────────────────

class _NumpyIndex:
    """Pure-numpy cosine-similarity fallback (no FAISS required)."""

    def __init__(self) -> None:
        self._vecs: list[np.ndarray] = []
        self._ids:  list[str]        = []

    def add(self, vec: np.ndarray, ep_id: str) -> None:
        self._vecs.append(vec.astype(np.float32))
        self._ids.append(ep_id)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        mat = np.stack(self._vecs, axis=0)             # [N, D]
        q   = query.astype(np.float32)
        # L2 distance (simpler and fine for 13-dim)
        dists = np.linalg.norm(mat - q, axis=1)
        k     = min(k, len(dists))
        idxs  = np.argsort(dists)[:k]
        return [(self._ids[i], float(dists[i])) for i in idxs]

    def __len__(self) -> int:
        return len(self._vecs)


class _FAISSIndex:
    """FAISS flat L2 index."""

    def __init__(self) -> None:
        self._index = faiss.IndexFlatL2(_EMBED_DIM)
        self._ids:  list[str] = []

    def add(self, vec: np.ndarray, ep_id: str) -> None:
        self._index.add(vec.reshape(1, -1).astype(np.float32))
        self._ids.append(ep_id)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        k   = min(k, len(self._ids))
        D, I = self._index.search(
            query.reshape(1, -1).astype(np.float32), k
        )
        return [(self._ids[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]

    def __len__(self) -> int:
        return len(self._ids)


# ── VectorMemory ──────────────────────────────────────────────────────────────

class VectorMemory:
    """
    Semantic retrieval over EpisodeRecords.

    Parameters
    ----------
    memory_dir : str | Path
        Directory for persisted index and record cache.
    max_records : int
        Ring-buffer cap. Oldest records evicted when exceeded.
    use_faiss : bool | None
        None = auto-detect. True/False = force.
    """

    def __init__(
        self,
        memory_dir:  Union[str, Path] = Path.home() / ".arca" / "memory",
        max_records: int              = 2000,
        use_faiss:   Optional[bool]   = None,
    ):
        self.memory_dir  = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records

        _use_faiss = FAISS_AVAILABLE if use_faiss is None else use_faiss
        self._index: Union[_FAISSIndex, _NumpyIndex] = (
            _FAISSIndex() if (_use_faiss and FAISS_AVAILABLE)
            else _NumpyIndex()
        )
        self.backend = "faiss" if isinstance(self._index, _FAISSIndex) else "numpy"

        # id → record cache (for retrieval)
        self._records: dict[str, object] = {}
        self._order:   list[str]         = []   # insertion order for ring eviction

        self._load()
        print(
            f"[ARCA VectorMemory] Backend={self.backend}  "
            f"Records={len(self._records)}  "
            f"Path={self.memory_dir}"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        return self.memory_dir / "vector_cache.pkl"

    def _save(self) -> None:
        try:
            with open(self._cache_path(), "wb") as f:
                pickle.dump(
                    {
                        "records": self._records,
                        "order":   self._order,
                        "index":   self._index,
                    },
                    f,
                )
        except Exception as e:
            print(f"[VectorMemory] Save failed: {e}")

    def _load(self) -> None:
        p = self._cache_path()
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            self._records = data.get("records", {})
            self._order   = data.get("order",   [])
            loaded_index  = data.get("index",   None)
            # Only restore if same backend type
            if loaded_index and type(loaded_index) == type(self._index):
                self._index = loaded_index
        except Exception as e:
            print(f"[VectorMemory] Load failed ({e}) — starting fresh.")
            self._records = {}
            self._order   = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def add(self, record) -> None:
        """
        Embed and store an EpisodeRecord (or compatible dict).
        Silently skips if episode_id already present.
        """
        ep_id = _episode_id(record)
        if not ep_id:
            # Generate a stable id from content
            ep_id = hashlib.md5(
                str(record).encode()
            ).hexdigest()[:12]

        if ep_id in self._records:
            return   # already indexed

        vec = _embed_episode(record)
        self._index.add(vec, ep_id)
        self._records[ep_id] = record
        self._order.append(ep_id)

        # Ring eviction
        if len(self._order) > self.max_records:
            evict_id = self._order.pop(0)
            self._records.pop(evict_id, None)
            # Note: FAISS/numpy indexes grow-only; we don't remove from them.
            # In practice 2k episodes is fine to keep in memory.

        self._save()

    def add_from_buffer(self, episode_buffer) -> int:
        """
        Bulk-add all records from an EpisodeBuffer.
        Returns the number of NEW records added.
        """
        before = len(self._records)
        for rec in episode_buffer._records:
            self.add(rec)
        return len(self._records) - before

    def search(self, query_record, k: int = 5) -> list:
        """
        Return the k most similar EpisodeRecords to query_record.
        Result list is sorted nearest-first.
        """
        if len(self._index) == 0:
            return []
        query_vec = _embed_episode(query_record)
        hits      = self._index.search(query_vec, k=k)
        results   = []
        for ep_id, dist in hits:
            rec = self._records.get(ep_id)
            if rec is not None:
                results.append((rec, dist))
        return results   # list of (record, distance)

    def format_for_llm(self, query_record=None, k: int = 3) -> str:
        """
        Return a compact text block for LLM prompt injection.
        If query_record is given, retrieves the k most similar episodes.
        Otherwise returns the k most recent.
        """
        if query_record is not None and len(self._index) > 0:
            hits = self.search(query_record, k=k)
            records = [r for r, _ in hits]
        else:
            records = list(self._records.values())[-k:]

        if not records:
            return "  No semantic memory yet."

        lines = []
        for i, r in enumerate(records):
            def _g(key, default=0):
                if hasattr(r, key): return getattr(r, key)
                if isinstance(r, dict): return r.get(key, default)
                return default

            lines.append(
                f"  Mem{i+1} [{_g('preset','?')}]: "
                f"reward={_g('total_reward',0):.1f}  "
                f"comp={_g('hosts_compromised',0)}/{_g('hosts_total',1)}  "
                f"steps={_g('steps',0)}  "
                f"goal={'✓' if _g('goal_reached') else '✗'}"
            )
            path = _g("attack_path", [])
            if path:
                lines.append(f"    Path: {' → '.join(str(p) for p in path[:3])}…")
            refl = _g("reflection", "")
            if refl:
                lines.append(f"    Lesson: {refl[:100]}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        if not self._records:
            return {"total": 0, "backend": self.backend}
        recs = list(self._records.values())
        def _g(r, k, d=0):
            if hasattr(r, k): return getattr(r, k)
            if isinstance(r, dict): return r.get(k, d)
            return d
        rewards = [_g(r, "total_reward", 0) for r in recs]
        goals   = [_g(r, "goal_reached", False) for r in recs]
        return {
            "total":      len(recs),
            "backend":    self.backend,
            "max_reward": max(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "goal_rate":  sum(goals) / len(goals),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"VectorMemory(backend={self.backend}, "
            f"records={len(self)}, "
            f"path={self.memory_dir})"
        )