"""Structured failure codes shared by the outbound tools, the Worker and the Coordinator (Round 4e / B-3, I-17).

Why this module exists: the Worker used to decide "did the tool fail?" by reading the tool's spoken text for words ("暂时连不上",
"unavailable", ...), and the Coordinator's weather-failure template accepted any short ASCII token in its parenthesis. Both are
text-reading rules. Now a tool says HOW it failed through ``toolmeta.note(failure=<code>)`` (or by returning ``{"ok": false, ...}``),
and the only thing a parenthesized slot may carry is a member of the closed enum below. No imports here on purpose.
"""
from __future__ import annotations

# ``toolmeta`` key ``failure``: the tool ran and did NOT produce usable evidence.
BUSY = "busy"                # our own in-flight cap refused (a refusal of ours, not an outage)
NETWORK = "network"          # every source was unreachable / timed out
NO_DATA = "no_data"          # sources answered, none knows the thing asked about
NO_RESULTS = "no_results"    # searched, found nothing usable
FAILURE_CODES = frozenset({BUSY, NETWORK, NO_DATA, NO_RESULTS})

# What weather._first_ok writes after "<source>:" in its error notes. Closed: a Worker cannot put free text in this slot (I-17).
SOURCE_ERROR_CODES = ("no_data", "timeout", "network", "http", "bad_data", "error")
