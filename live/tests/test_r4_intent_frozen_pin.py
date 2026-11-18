"""The frozen held-out set (tests/intent_frozen.py) must not change: its sha256 is pinned in tests/intent_frozen.sha256."""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

_T = Path(__file__).resolve().parent


class FrozenSetIsFrozen(unittest.TestCase):
    def test_content_matches_the_pinned_hash(self) -> None:
        want = (_T / "intent_frozen.sha256").read_text().strip()
        got = hashlib.sha256((_T / "intent_frozen.py").read_bytes()).hexdigest()
        self.assertEqual(got, want, "tests/intent_frozen.py was edited; it is a frozen held-out set (Critic I-10). Add a NEW set instead.")

    def test_size(self) -> None:
        from tests import intent_frozen as f
        self.assertEqual((len(f.FROZEN_REALTIME), len(f.FROZEN_FREE)), (57, 61))
        self.assertEqual(len(set(f.FROZEN_REALTIME + f.FROZEN_FREE)), 118)


if __name__ == "__main__":
    unittest.main()
