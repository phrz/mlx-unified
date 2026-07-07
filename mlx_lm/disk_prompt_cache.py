# mlx-unified: disk spill tier for LRUPromptCache.
#
# When the in-memory prompt cache evicts an entry (count or byte budget), the
# entry is handed here instead of dropped: a background writer serializes it to
# a temporary safetensors file, and a later request that shares a prefix loads
# it back (move semantics — a loaded file is deleted; re-eviction re-spills).
# The concept comes from LM Studio's mlx-engine disk prompt cache; the
# implementation is native to mlx-lm's PromptTrie/LRU design and stores whole
# entries rather than per-chunk deltas.
#
# Threading: the generation thread calls plan()/load()/offer(); the writer
# thread commits saves and evictions. All index mutations happen under one
# lock; file I/O happens outside it (each file has a single owner at any time:
# the writer until the index entry is published, then whoever pops the entry).
# An entry queued for save is invisible to plan() until its write commits — a
# fetch racing a spill just misses and recomputes.

import atexit
import logging
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from queue import Queue
from typing import Any, List, Optional

import mlx.core as mx

from .models.cache import (
    PromptTrie,
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class _DiskEntry:
    path: Path
    nbytes: int
    cache_type: str
    trimmable: bool


@dataclass
class DiskRestorePlan:
    key_tokens: List[int]  # the stored entry's token key
    prefix_len: int  # tokens of the request this entry can serve
    trim: int  # tokens to trim after load (longer-entry case)


class DiskPromptCacheStore:
    def __init__(self, max_bytes: int, directory: Optional[str] = None):
        self.max_bytes = max_bytes
        if directory is None:
            self._dir = Path(tempfile.mkdtemp(prefix="mlx_lm_prompt_cache_"))
            atexit.register(shutil.rmtree, self._dir, ignore_errors=True)
        else:
            self._dir = Path(directory)
            self._dir.mkdir(parents=True, exist_ok=True)
        self._trie = PromptTrie()
        self._lru: OrderedDict = OrderedDict()  # (model, tuple(tokens)) -> None
        self._n_bytes = 0
        self._seq = count()
        self._lock = threading.Lock()
        self._queue: Queue = Queue()
        self._writer = threading.Thread(
            target=self._write_loop, name="mlx-lm-prompt-cache-disk", daemon=True
        )
        self._writer.start()
        logger.info(
            "Disk prompt cache: dir=%s budget=%.1f GiB (temporary, "
            "cleared on process exit)",
            self._dir,
            max_bytes / (1 << 30),
        )

    @property
    def nbytes(self) -> int:
        return self._n_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._lru)

    def offer(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        approx_nbytes: int,
        cache_type: str,
    ) -> None:
        """Accept an evicted entry for spilling (called on the generation
        thread; serialization happens on the writer thread)."""
        if approx_nbytes > self.max_bytes:
            return
        # Materialize on the owning thread — the writer thread has no claim on
        # this thread's GPU stream and must only serialize settled arrays.
        mx.eval([c.state for c in prompt_cache])
        self._queue.put(
            (model, list(tokens), prompt_cache, cache_type, can_trim_prompt_cache(prompt_cache))
        )

    def plan(self, model: Any, tokens: List[int]) -> Optional[DiskRestorePlan]:
        """Best disk prefix for this prompt, mirroring LRUPromptCache's
        selection (exact → trimmable-longer → shorter). No I/O."""
        with self._lock:
            result = self._trie.search(model, tokens)
            if result.exact is not None:
                return DiskRestorePlan(result.exact, len(result.exact), 0)
            short_length = len(result.shorter) if result.shorter is not None else 0
            if result.longer is not None and result.common_prefix > short_length:
                entry = self._trie.get(model, result.longer)
                if entry.trimmable:
                    prefix = min(len(tokens) - 1, result.common_prefix)
                    return DiskRestorePlan(
                        result.longer, prefix, len(result.longer) - prefix
                    )
            if short_length > 0:
                return DiskRestorePlan(result.shorter, short_length, 0)
        return None

    def load(self, model: Any, plan: DiskRestorePlan) -> Optional[List[Any]]:
        """Pop the planned entry and load its cache (move semantics — the file
        is deleted; the caller owns the arrays). None on any failure."""
        with self._lock:
            try:
                entry = self._pop_locked(model, plan.key_tokens)
            except KeyError:
                return None
        try:
            cache = load_prompt_cache(str(entry.path))
        except Exception:
            logger.warning(
                "Disk prompt cache load failed for %s; dropping entry.",
                entry.path,
                exc_info=True,
            )
            cache = None
        entry.path.unlink(missing_ok=True)
        return cache

    def close(self) -> None:
        self._queue.put(None)
        self._writer.join(timeout=5)
        shutil.rmtree(self._dir, ignore_errors=True)

    def _pop_locked(self, model: Any, tokens: List[int]) -> _DiskEntry:
        entry = self._trie.pop(model, tokens)
        self._lru.pop((model, tuple(tokens)), None)
        self._n_bytes -= entry.nbytes
        return entry

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            model, tokens, prompt_cache, cache_type, trimmable = item
            path = self._dir / f"kv_{next(self._seq)}.safetensors"
            try:
                # This worker thread owns no GPU stream; state-slicing and the
                # safetensors write run on the CPU stream (unified memory — the
                # arrays were materialized by offer() on the owning thread).
                with mx.stream(mx.cpu):
                    save_prompt_cache(
                        str(path), prompt_cache, {"cache_type": cache_type}
                    )
                nbytes = path.stat().st_size
            except Exception:
                logger.warning(
                    "Disk prompt cache save failed; dropping entry.", exc_info=True
                )
                path.unlink(missing_ok=True)
                continue
            # The cache arrays only stay alive through this loop iteration —
            # drop our reference before publishing so spilled entries never
            # hold GPU memory past their write.
            del prompt_cache

            evicted: List[_DiskEntry] = []
            with self._lock:
                key = (model, tuple(tokens))
                if key in self._lru:
                    # A newer spill of the same tokens supersedes the old file.
                    evicted.append(self._pop_locked(model, tokens))
                self._trie.add(model, tokens, _DiskEntry(path, nbytes, cache_type, trimmable))
                self._lru[key] = None
                self._n_bytes += nbytes
                while self._n_bytes > self.max_bytes and len(self._lru) > 1:
                    old_model, old_tokens = next(iter(self._lru))
                    evicted.append(self._pop_locked(old_model, list(old_tokens)))
            for entry in evicted:
                entry.path.unlink(missing_ok=True)
