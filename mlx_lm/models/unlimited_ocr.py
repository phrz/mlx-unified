# Copyright © 2026 Apple Inc.
#
# Unlimited-OCR: the deepseekocr text body plus R-SWA decoding — the full
# prefill KV is retained forever while decode tokens beyond a small window
# rotate through a fixed ring of slots. Ported from mlx-vlm's
# models/unlimited_ocr/language.py (MIT © Blaizzy / mlx-vlm contributors).

from dataclasses import dataclass
from typing import Optional

from . import deepseekocr
from .cache import KVCache


@dataclass
class ModelArgs(deepseekocr.ModelArgs):
    model_type: str = "unlimited_ocr"
    sliding_window_size: Optional[int] = 128
    sliding_window: Optional[int] = None


class RingSlidingKVCache(KVCache):
    """R-SWA cache: a standard causal cache through prefill and the first
    `window_size` decode tokens, after which new decode keys/values overwrite
    that ring in place while `offset` (the absolute position, and therefore
    the RoPE offset) keeps increasing."""

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.prefill_length: Optional[int] = None
        self._ring_pos = 0

    def update_and_fetch(self, keys, values):
        seq_len = keys.shape[2]

        if self.prefill_length is None:
            # Prefill chunks (seq_len > 1) append normally; the first 1-token
            # decode step fixes the retained-prompt boundary.
            if seq_len > 1:
                return super().update_and_fetch(keys, values)
            self.prefill_length = self.offset

        if self.keys is None or self.offset < self.prefill_length + self.window_size:
            keys, values = super().update_and_fetch(keys, values)
            if self.offset >= self.prefill_length + self.window_size:
                self._ring_pos = 0
            return keys, values

        for idx in range(seq_len):
            slot = self.prefill_length + self._ring_pos
            self.keys[..., slot : slot + 1, :] = keys[..., idx : idx + 1, :]
            self.values[..., slot : slot + 1, :] = values[..., idx : idx + 1, :]
            self._ring_pos = (self._ring_pos + 1) % self.window_size

        self.offset += seq_len
        end = self.prefill_length + self.window_size
        return self.keys[..., :end, :], self.values[..., :end, :]

    def make_mask(self, N, return_array=False, window_size=None):
        # The ring itself enforces the window, so never build a windowed mask.
        # Steady-state decode attends mask-free over the retained prefill plus
        # every ring slot (ring order is not causal order).
        if (
            self.prefill_length is not None
            and self.offset >= self.prefill_length + self.window_size
            and N == 1
            and not return_array
        ):
            return None
        return super().make_mask(N, return_array=return_array, window_size=None)

    def is_trimmable(self):
        # Ring slots stop corresponding to trailing positions once decode begins.
        return self.prefill_length is None

    @property
    def state(self):
        if self.keys is None:
            return None, None
        end = (
            self.offset
            if self.prefill_length is None
            else min(self.offset, self.prefill_length + self.window_size)
        )
        return self.keys[..., :end, :], self.values[..., :end, :]

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = 0 if self.keys is None else self.keys.shape[2]
        self.prefill_length = None
        self._ring_pos = 0

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.window_size,
                    -1 if self.prefill_length is None else self.prefill_length,
                    self.offset,
                    self._ring_pos,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        window_size, prefill_length, offset, ring_pos = map(int, v)
        self.window_size = window_size
        self.prefill_length = None if prefill_length < 0 else prefill_length
        self.offset = offset
        self._ring_pos = ring_pos


class Model(deepseekocr.Model):
    def make_cache(self):
        window_size = self.args.sliding_window_size or self.args.sliding_window
        if window_size is None:
            return [KVCache() for _ in self.layers]
        return [RingSlidingKVCache(window_size) for _ in self.layers]
