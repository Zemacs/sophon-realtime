"""Round 4d / S-25: the gateway binds loopback unless told otherwise (it has no authentication of its own)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from sophon_live import gateway


class DefaultBindAddress(unittest.TestCase):
    def _host(self, env: dict[str, str], argv: list[str]) -> str:
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("SOPHON_LIVE_HOST", None) if "SOPHON_LIVE_HOST" not in env else None
            return gateway._build_parser().parse_args(argv).host

    def test_default_is_loopback(self) -> None:
        self.assertEqual(self._host({}, []), "127.0.0.1")
        self.assertEqual(self._host({"SOPHON_LIVE_HOST": ""}, []), "127.0.0.1")
        self.assertEqual(self._host({"SOPHON_LIVE_HOST": "  "}, []), "127.0.0.1")

    def test_explicit_choices_win(self) -> None:
        self.assertEqual(self._host({"SOPHON_LIVE_HOST": "0.0.0.0"}, []), "0.0.0.0")
        self.assertEqual(self._host({}, ["--host", "0.0.0.0"]), "0.0.0.0")
        self.assertEqual(self._host({"SOPHON_LIVE_HOST": "0.0.0.0"}, ["--host", "127.0.0.1"]), "127.0.0.1")

    def test_other_defaults_are_unchanged(self) -> None:
        a = gateway._build_parser().parse_args([])
        self.assertEqual((a.port, a.path), (8780, "/v1/live/sessions"))

    def test_startup_line_says_who_can_connect(self) -> None:
        self.assertIn("loopback", gateway._exposure_note("127.0.0.1"))
        wide = gateway._exposure_note("0.0.0.0")
        self.assertIn("NO authentication", wide)
        self.assertIn("0.0.0.0", wide)


if __name__ == "__main__":
    unittest.main()
