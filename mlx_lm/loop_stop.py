# Copyright © 2026 Apple Inc.
"""
mlx-unified: stop a reply that has fallen into a verbatim loop.

A greedy OCR reader can write the same stretch forever (an empty table row,
``<tr><td></td>…</tr>``, hundreds of times) until it hits ``max_tokens``.
The request opts in with ``x_loop_stop``; the server then checks the reply's
tail as it grows and ends it with ``finish_reason: "loop"`` once one stretch
has repeated back to back ``min_repeats`` times, its repeats spanning at least
``min_chars``. Everything written before the loop is kept; the caller decides
what to do with the repeats.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LoopStopOptions:
    # Back-to-back copies of one stretch that count as a loop. A real blank form
    # repeats an empty row a few times; a runaway repeats it hundreds of times.
    min_repeats: int = 12
    # The copies must span at least this many characters, so a stretch of a few
    # characters (a run of dashes, of spaces) is not a loop until it is long.
    min_chars: int = 600
    # The longest stretch looked for.
    max_period: int = 2000
    # Check after at least this many new characters.
    every_chars: int = 32


def parse_loop_stop(value: Any) -> Optional[LoopStopOptions]:
    """`true` takes the defaults; an object overrides them; anything else is off."""
    if value is True:
        return LoopStopOptions()
    if isinstance(value, dict):
        options = LoopStopOptions()
        for key in ("min_repeats", "min_chars", "max_period", "every_chars"):
            if key in value:
                number = value[key]
                if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                    raise ValueError(f"x_loop_stop.{key} must be a positive integer")
                setattr(options, key, number)
        options.min_repeats = max(2, options.min_repeats)
        return options
    if value in (None, False):
        return None
    raise ValueError("x_loop_stop must be true or an object")


def repeated_tail(text: str, options: LoopStopOptions) -> Optional[int]:
    """The period of the loop the text ends in, or None.

    Candidates come from earlier occurrences of the text's last few characters,
    nearest first, so the shortest repeating stretch is found first.
    """
    n = len(text)
    probe = min(32, max(1, n // options.min_repeats))
    if n < options.min_chars or n < probe * options.min_repeats:
        return None
    window = text[n - probe :]
    search_end = n - 1
    tried = 0
    while tried < 64:
        i = text.rfind(window, 0, search_end)
        if i < 0:
            return None
        period = n - probe - i
        if period > options.max_period:
            return None
        search_end = i + probe - 1
        tried += 1
        if period < 1:
            continue
        copies = max(options.min_repeats, -(-options.min_chars // period))
        if copies * period > n:
            continue
        block = text[n - period :]
        if all(text[n - k * period : n - (k - 1) * period] == block for k in range(2, copies + 1)):
            return period
    return None


class LoopStop:
    """Accumulates a reply's text and says when it has fallen into a loop."""

    def __init__(self, options: LoopStopOptions):
        self.options = options
        self.text = ""
        self._checked = 0

    def feed(self, piece: str) -> bool:
        self.text += piece
        if len(self.text) - self._checked < self.options.every_chars:
            return False
        self._checked = len(self.text)
        return repeated_tail(self.text, self.options) is not None
