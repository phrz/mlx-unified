# Copyright © 2026 Apple Inc.
#
# mlx-unified: grammar-enforced structured output for mlx_lm.server. A chat
# completion's `response_format` (`json_schema` or `json_object`) compiles to a
# regex with outlines-core, then to a token-level automaton over the served
# tokenizer's vocabulary. GuidedLogitsProcessor masks every token the automaton
# does not allow at the current state, so the sampled text is valid JSON that
# matches the schema (or the request ends on `max_tokens` with a truncated
# document, the way OpenAI reports `finish_reason: "length"`).
#
# outlines-core is optional (`pip install 'mlx-lm[structured]'`). Without it a
# request that carries a JSON response_format is rejected with a 400, never
# silently generated unconstrained.

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional dependency; see module docstring
    from outlines_core import Guide, Index, Vocabulary
    from outlines_core.json_schema import build_regex_from_schema

    OUTLINES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    Guide = Index = Vocabulary = None  # type: ignore[assignment]
    build_regex_from_schema = None  # type: ignore[assignment]
    OUTLINES_AVAILABLE = False

INSTALL_HINT = "install the structured extra: pip install 'mlx-lm[structured]'"


class StructuredOutputError(ValueError):
    """A response_format the server cannot enforce. The API maps it to 400."""


@dataclass(frozen=True)
class ResponseFormatSpec:
    kind: str  # "json_schema" | "json_object"
    regex: str
    name: Optional[str] = None


def parse_response_format(value: Any) -> Optional[ResponseFormatSpec]:
    """Validate an OpenAI `response_format` and compile its schema to a regex.

    Returns None for absent/`text` formats. Raises StructuredOutputError with a
    client-readable message for a malformed or unsupported value.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise StructuredOutputError("response_format must be an object")
    kind = value.get("type")
    if kind == "text":
        return None
    name = None
    if kind == "json_object":
        schema: Dict[str, Any] = {"type": "object"}
    elif kind == "json_schema":
        block = value.get("json_schema")
        if not isinstance(block, dict):
            raise StructuredOutputError("response_format.json_schema must be an object")
        schema = block.get("schema")
        if not isinstance(schema, dict):
            raise StructuredOutputError(
                "response_format.json_schema.schema must be a JSON schema object"
            )
        name = block.get("name")
        if name is not None and not isinstance(name, str):
            raise StructuredOutputError("response_format.json_schema.name must be a string")
        strict = block.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise StructuredOutputError("response_format.json_schema.strict must be a boolean")
    else:
        raise StructuredOutputError(
            f"unsupported response_format.type {kind!r}; "
            "expected text, json_object, or json_schema"
        )
    if not OUTLINES_AVAILABLE:
        raise StructuredOutputError(
            f"structured output ({kind}) requires outlines-core; {INSTALL_HINT}"
        )
    try:
        regex = build_regex_from_schema(json.dumps(schema))
    except Exception as e:  # outlines-core raises ValueError/TypeError/RuntimeError
        raise StructuredOutputError(f"unsupported {kind} schema: {e}") from e
    return ResponseFormatSpec(kind=kind, regex=regex, name=name)


# --- Vocabulary -----------------------------------------------------------------


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte-level BPE alphabet (the `Ġ`/`Ċ` mapping)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_UNICODE_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}


def token_bytes(token: str, byte_level: bool) -> bytes:
    """The raw bytes a vocabulary entry emits when decoded on its own."""
    if byte_level:
        try:
            return bytes(_UNICODE_TO_BYTE[c] for c in token)
        except KeyError:
            pass
    if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        try:
            return bytes([int(token[3:5], 16)])
        except ValueError:
            pass
    return token.replace("▁", " ").encode("utf-8")


def is_byte_level_vocab(vocab: Dict[str, int]) -> bool:
    return "Ġ" in vocab or "Ċ" in vocab  # Ġ / Ċ


def build_vocabulary(tokenizer, eos_token_id: int):
    """An outlines-core Vocabulary for a served tokenizer (wrapper or HF)."""
    if not OUTLINES_AVAILABLE:
        raise StructuredOutputError(f"outlines-core is not installed; {INSTALL_HINT}")
    vocab = tokenizer.get_vocab()
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    special.add(eos_token_id)
    byte_level = is_byte_level_vocab(vocab)
    mapping: Dict[bytes, List[int]] = {}
    for token, token_id in vocab.items():
        if token_id in special:
            continue
        raw = token_bytes(token, byte_level)
        if not raw:
            continue
        mapping.setdefault(raw, []).append(int(token_id))
    return Vocabulary(int(eos_token_id), mapping), max(int(t) for t in vocab.values())


class StructuredIndexCache:
    """Vocabulary per served model plus a bounded LRU of compiled indexes.

    Index compilation walks the whole vocabulary and can take seconds for a
    large schema; every request with the same schema on the same model reuses it.
    """

    def __init__(self, max_entries: int = 16):
        self._max_entries = max_entries
        self._vocab_key = None
        self._vocab = None
        self._max_token_id = 0
        self._indexes: "OrderedDict[Tuple[Any, str], Any]" = OrderedDict()

    def index_for(self, model_key, tokenizer, spec: ResponseFormatSpec):
        if self._vocab_key != model_key or self._vocab is None:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                raise StructuredOutputError(
                    "structured output needs a tokenizer with an eos token"
                )
            self._vocab, self._max_token_id = build_vocabulary(tokenizer, int(eos))
            self._vocab_key = model_key
            self._indexes.clear()
        key = (model_key, spec.regex)
        index = self._indexes.get(key)
        if index is not None:
            self._indexes.move_to_end(key)
            return index, self._max_token_id
        try:
            index = Index(spec.regex, self._vocab)
        except Exception as e:
            raise StructuredOutputError(
                f"cannot compile {spec.kind} schema for this tokenizer: {e}"
            ) from e
        self._indexes[key] = index
        while len(self._indexes) > self._max_entries:
            self._indexes.popitem(last=False)
        return index, self._max_token_id


# --- Logits processor -----------------------------------------------------------


def _endswith(recent: Sequence[int], suffix: Sequence[int]) -> bool:
    n = len(suffix)
    return n > 0 and len(recent) >= n and tuple(recent[-n:]) == tuple(suffix)


class GuidedLogitsProcessor:
    """`(tokens, logits) -> logits` for mlx-lm/mlx-vlm generation loops.

    The loops call a processor exactly once per generated token, with `tokens`
    ending in the token sampled at the previous step (the first call sees only
    prompt tokens). The guide advances on that last token, then the logits get
    `-inf` on every token the automaton rejects at the new state. Once the
    document is complete only the eos tokens stay allowed.

    Thinking models: an `initial_state` of "reasoning" (the prompt ends inside
    an open think block) leaves logits untouched until the think-end tokens are
    generated. From "normal", the first mask also allows the think-start token
    so a model that reasons first can still do so; the constraint engages after
    the block closes.
    """

    def __init__(
        self,
        index,
        eos_token_ids: Iterable[int],
        *,
        max_token_id: int = 0,
        initial_state: str = "normal",
        think_start_tokens: Sequence[int] = (),
        think_end_tokens: Sequence[int] = (),
        mask_cache_size: int = 512,
    ):
        if not OUTLINES_AVAILABLE:
            raise StructuredOutputError(f"outlines-core is not installed; {INSTALL_HINT}")
        self._guide = Guide(index)
        self._eos = tuple(int(t) for t in eos_token_ids)
        if not self._eos:
            raise StructuredOutputError("structured output needs at least one eos token")
        self._max_token_id = int(max_token_id)
        self._think_start = tuple(int(t) for t in think_start_tokens)
        self._think_end = tuple(int(t) for t in think_end_tokens)
        thinking = bool(self._think_start) and bool(self._think_end)
        self._phase = "reasoning" if (initial_state == "reasoning" and thinking) else "guided"
        # From a normal start the model may still open a think block first.
        self._think_offer = thinking and self._phase == "guided"
        self._think_progress = 0
        self._calls = 0
        self._broken = False
        self._recent: List[int] = []
        self._mask_cache: "OrderedDict[Any, Any]" = OrderedDict()
        self._mask_cache_size = mask_cache_size

    # -- state --

    @property
    def phase(self) -> str:
        return self._phase

    def observe(self, token: int) -> None:
        """Feed the token sampled at the previous step."""
        token = int(token)
        self._recent.append(token)
        if len(self._recent) > 64:
            del self._recent[:-64]
        if self._phase == "reasoning":
            if _endswith(self._recent, self._think_end):
                self._phase = "guided"
            return
        if self._phase == "done" or self._broken:
            return
        if self._think_offer:
            if token == self._think_start[self._think_progress]:
                self._think_progress += 1
                if self._think_progress == len(self._think_start):
                    self._phase = "reasoning"
                    self._think_offer = False
                    self._think_progress = 0
                return
            self._think_offer = False
        if self._guide.is_finished():
            self._phase = "done"
            return
        try:
            self._guide.advance(token, False)
        except Exception as e:  # a token the automaton never allowed
            logging.warning("structured output: token %d left the grammar (%s)", token, e)
            self._broken = True

    def __call__(self, tokens, logits):
        if self._calls > 0 and len(tokens) > 0:
            self.observe(int(tokens[-1].item()))
        self._calls += 1
        if self._phase == "reasoning":
            return logits
        return logits + self.mask(int(logits.shape[-1]), logits.dtype)

    # -- masks --

    def allowed_token_ids(self, vocab_size: int) -> np.ndarray:
        """Boolean vector: which of `vocab_size` logits stay allowed right now."""
        allowed = np.zeros(vocab_size, dtype=bool)
        if self._phase == "done" or self._broken or self._guide.is_finished():
            for t in self._eos:
                if t < vocab_size:
                    allowed[t] = True
            return allowed
        words = (max(vocab_size, self._max_token_id + 1) + 31) // 32
        buf = np.zeros(words, dtype=np.uint32)
        try:
            self._guide.write_mask_into(buf.ctypes.data, buf.size, buf.itemsize)
            bits = np.unpackbits(buf.view(np.uint8), bitorder="little")[:vocab_size]
            allowed |= bits.astype(bool)
        except AttributeError:  # older outlines-core: no bitmask writer
            ids = np.fromiter(self._guide.get_tokens(), dtype=np.int64)
            allowed[ids[ids < vocab_size]] = True
        if self._think_offer and self._think_progress < len(self._think_start):
            t = self._think_start[self._think_progress]
            if t < vocab_size:
                allowed[t] = True
        return allowed

    def mask(self, vocab_size: int, dtype=None):
        """An additive mask (0 or -inf) as an mx.array, cached per guide state."""
        import mlx.core as mx

        offering = self._think_offer and self._think_progress < len(self._think_start)
        terminal = self._phase == "done" or self._broken or self._guide.is_finished()
        key = ("eos", vocab_size) if terminal else (self._guide.get_state(), vocab_size, offering)
        cached = self._mask_cache.get(key)
        if cached is not None:
            self._mask_cache.move_to_end(key)
            return cached
        allowed = self.allowed_token_ids(vocab_size)
        mask = mx.array(np.where(allowed, 0.0, -np.inf).astype(np.float32))
        self._mask_cache[key] = mask
        while len(self._mask_cache) > self._mask_cache_size:
            self._mask_cache.popitem(last=False)
        return mask


def make_guided_processor(
    cache: StructuredIndexCache,
    model_key,
    tokenizer,
    spec: Optional[ResponseFormatSpec],
    initial_state: str = "normal",
) -> List[GuidedLogitsProcessor]:
    """The extra logits processors a request needs (empty without a format)."""
    if spec is None:
        return []
    index, max_token_id = cache.index_for(model_key, tokenizer, spec)
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if not eos_ids:
        eos_ids = {tokenizer.eos_token_id}
    has_thinking = bool(getattr(tokenizer, "has_thinking", False))
    return [
        GuidedLogitsProcessor(
            index,
            eos_ids,
            max_token_id=max_token_id,
            initial_state=initial_state,
            think_start_tokens=(
                getattr(tokenizer, "_think_start_tokens", ()) if has_thinking else ()
            ),
            think_end_tokens=(
                getattr(tokenizer, "_think_end_tokens", ()) if has_thinking else ()
            ),
        )
    ]
