"""Round 4i / I-22: the test page has no demo backend (it injected a fake answer next to the real result when the backend was coord);
and a retired env var that is still set is announced once at startup (name only)."""
from __future__ import annotations

import os
import re
import sys
import unittest
from unittest import mock

from sophon_live import gateway

PAGE = os.path.join(os.path.dirname(os.path.abspath(gateway.__file__)), "static", "index.html")


def _page() -> str:
    return open(PAGE, encoding="utf-8").read()


def _on_delegation_body(html: str) -> str:
    start = html.index("async function onDelegation")
    return html[start:html.index("function inject(", start)]


class TestePageHasNoDemoBackend(unittest.TestCase):
    """The gateway serves this very file at GET / and /index.html, so what is in it is what ships."""

    def test_no_demo_backend_code_is_in_the_served_page(self) -> None:
        html = _page()
        for needle in ("demoBackend", "CITIES", "async function weather", "open-meteo", "simulated device", "模拟设备", "内置演示后端"):
            self.assertNotIn(needle, html, needle)
        self.assertNotIn("?demo", html)

    def test_the_page_makes_no_network_request_of_its_own(self) -> None:
        self.assertNotIn("fetch(", _page())

    def test_gateway_side_backends_are_display_only(self) -> None:
        """For backend orch AND coord the delegation handler returns before anything is injected from the page."""
        body = _on_delegation_body(_page())
        m = re.search(r'if \(d\.backend === "orch" \|\| d\.backend === "coord"\) \{[^}]*return;', body)
        self.assertIsNotNone(m, body)
        self.assertNotIn("inject(", body[:m.end()])

    def test_the_page_never_answers_a_delegation_by_itself(self) -> None:
        """Whatever the backend, the only way the page injects an answer is the user's own manual box."""
        body = _on_delegation_body(_page())
        self.assertNotIn("inject(", body)
        self.assertNotIn("setTimeout", body)
        self.assertIn("pendingDeleg", body)

    def test_the_served_file_is_this_file(self) -> None:
        src = open(gateway.__file__, encoding="utf-8").read()
        self.assertIn('"static", "index.html"', src)


class RetiredEnvWarning(unittest.TestCase):
    def test_set_variable_warns_once_by_name_without_echoing_the_value(self) -> None:
        with mock.patch.dict(os.environ, {"SOPHON_LIVE_GROUND_MODE": "sekrit-value-123"}):
            with self.assertLogs("sophon_live", level="WARNING") as cm:
                hit = gateway.warn_retired_env()
        self.assertEqual(hit, ["SOPHON_LIVE_GROUND_MODE"])
        self.assertEqual(len(cm.records), 1)
        text = cm.output[0]
        self.assertIn("SOPHON_LIVE_GROUND_MODE", text)
        self.assertIn("no longer used", text)
        self.assertIn("no effect", text)
        self.assertNotIn("sekrit-value-123", text)

    def test_empty_value_still_counts_as_set(self) -> None:
        with mock.patch.dict(os.environ, {"SOPHON_LIVE_GROUND_MODE": ""}):
            with self.assertLogs("sophon_live", level="WARNING"):
                self.assertEqual(gateway.warn_retired_env(), ["SOPHON_LIVE_GROUND_MODE"])

    def test_unset_variable_is_silent(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "SOPHON_LIVE_GROUND_MODE"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertNoLogs("sophon_live", level="WARNING"):
                self.assertEqual(gateway.warn_retired_env(), [])

    def test_main_runs_the_check_at_startup(self) -> None:
        def _drop(coro):
            coro.close()
        with mock.patch.dict(os.environ, {"SOPHON_LIVE_GROUND_MODE": "placeholder"}), \
             mock.patch.object(sys, "argv", ["sophon-live"]), \
             mock.patch.object(gateway.asyncio, "run", _drop), \
             mock.patch.object(gateway.logging, "basicConfig"), \
             mock.patch.object(gateway, "ASR", lambda path: (lambda audio: "")), \
             self.assertLogs("sophon_live", level="WARNING") as cm:
            gateway.main()
        self.assertEqual(sum("SOPHON_LIVE_GROUND_MODE" in m for m in cm.output), 1, cm.output)


if __name__ == "__main__":
    unittest.main()
