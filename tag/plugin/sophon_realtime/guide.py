"""Guidance entry (bridge POST /bridge/<key>/guide): the front receives a guidance payload (contracts/schemas/front/guidance.schema.json),
checks its structure only, and puts its facts into the context of the next generated reply.

Structure only: field presence, JSON types, closed enums, id character sets and length limits -- the same things the schema says.
Free text (fact values, units) is copied into the context as given and never read; nothing here branches on what a string says.
The model never writes ids: guidance_id and version ride on the request (info["guidance_ref"]) and on the history entry.
Pure Python (no vllm import) so it can be tested without a GPU."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime

_ALNUM = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_ID_REST = _ALNUM | frozenset("._:-")
_DIGITS = frozenset("0123456789")
_HEX = frozenset("0123456789abcdef")

TOP_REQUIRED = "schema_version guidance_id version session_id emitted_at readout_id snapshot_last_entry_id turn_kind release seq run_fingerprint host_id".split()
TOP_OPTIONAL = ["current_turn", "opened_tasks", "disposition_facts"]
TURN_KINDS = ["user_turn", "event_turn", "timer_turn"]
H1 = ["to_system", "feedback", "not_to_system", "holding"]
H4 = ["immediate", "after_playback", "next_user_turn", "only_when_asked"]
H7 = ["silent", "answer", "ask"]
H8 = ["normal", "refuse", "escalate"]
ITEM_REQUIRED = "item_id item_version kind released_by facts".split()
ITEM_KINDS = "result progress failure playback_remainder reminder_due question".split()
NEEDS_EVIDENCE = ["result", "failure", "progress"]
CONTEXT_KEYS = "guidance_id version turn_kind current_turn opened_tasks release disposition_facts".split()
SYS_GUIDE = {
    "zh": "对话里 tool 角色的 JSON 是执行中枢给你的指导：只含结构化事实，不含现成的句子。请据此用自然口语说给用户，不要念字段名、编号或 JSON，也不要编造指导里没有的事实。",
    "en": "A tool-role JSON in the conversation is guidance from the execution hub: structured facts only, no ready-made sentences. Say it to the user in natural spoken language; do not read out field names, ids or JSON, and do not invent facts that are not in it.",
}


def _is_id(v):
    ok = isinstance(v, str) and 1 <= len(v) <= 64 and v[0] in _ALNUM
    return ok and all(c in _ID_REST for c in v)


def _is_uint(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _is_semver(v):
    parts = v.split(".") if isinstance(v, str) else []
    return len(parts) == 3 and all(p and all(c in _DIGITS for c in p) for p in parts)


def _is_hex64(v):
    return isinstance(v, str) and len(v) == 64 and all(c in _HEX for c in v)


def _is_time(v):
    try:
        t = datetime.fromisoformat(v[:-1] + "+00:00" if v.endswith("Z") else v)
    except (AttributeError, ValueError):
        return False
    return t.tzinfo is not None


def _is_scalar(v):
    short = isinstance(v, str) and len(v) <= 64
    return v is None or isinstance(v, (int, float)) or short


_ID = (_is_id, "bad id")
_UINT = (_is_uint, "must be a non-negative integer")


def _obj(v, path, required, optional=()):
    """Errors for a non-object, missing required keys and keys the schema does not allow."""
    if not isinstance(v, dict):
        return [f"{path}: must be an object"]
    allowed = [*required, *optional]
    return [f"{path}.{k}: required" for k in required if k not in v] + [
        f"{path}.{k}: not allowed" for k in v if k not in allowed
    ]


def _check(obj, spec, path):
    """Errors for the keys present in obj whose value fails its (test, message)."""
    bad = [k for k, (ok, _) in spec.items() if k in obj and not ok(obj[k])]
    return [f"{path}.{k}: {spec[k][1]}" for k in bad]


def _enum(obj, key, path, options):
    ok = key not in obj or (isinstance(obj[key], str) and obj[key] in options)
    return [] if ok else [f"{path}.{key}: must be one of {'/'.join(options)}"]


def _each(v, path, check):
    """Errors from check(item, item_path) for every item of an array; a non-array is one error."""
    if not isinstance(v, list):
        return [f"{path}: must be an array"]
    return [e for i, x in enumerate(v) for e in check(x, f"{path}[{i}]")]


def _fact(f, path):
    errs = _obj(f, path, ["key", "value"], ["unit"])
    if isinstance(f, dict):
        errs += _check(f, {"key": _ID, "unit": _ID}, path)
        if "value" in f and not _is_scalar(f["value"]):
            errs.append(f"{path}.value: must be a scalar of at most 64 chars")
    return errs


def _ref(r, path):
    errs = _obj(r, path, ["id", "version"])
    return errs or _check(r, {"id": _ID, "version": _UINT}, path)


def _item(it, path):
    errs = _obj(it, path, ITEM_REQUIRED, ["evidence"])
    if not isinstance(it, dict):
        return errs
    errs += _check(it, {"item_id": _ID, "item_version": _UINT}, path)
    errs += _enum(it, "kind", path, ITEM_KINDS) + _enum(it, "released_by", path, H4)
    errs += _each(it.get("facts", []), f"{path}.facts", _fact)
    if "evidence" in it:
        errs += _each(it["evidence"], f"{path}.evidence", _ref)
    if it.get("kind") in NEEDS_EVIDENCE and not it.get("evidence"):
        errs.append(f"{path}.evidence: at least one reference is required")
    return errs


def _task(t, path):
    errs = _obj(t, path, ["task_id", "task_version"], ["acceptance_event_id"])
    spec = {"task_id": _ID, "acceptance_event_id": _ID, "task_version": _UINT}
    return errs + (_check(t, spec, path) if isinstance(t, dict) else [])


def _turn(ct, path):
    errs = _obj(ct, path, ["H1", "H7", "H8"])
    if isinstance(ct, dict):
        for k, options in (("H1", H1), ("H7", H7), ("H8", H8)):
            errs += _enum(ct, k, path, options)
    return errs


def validate(g) -> list[str]:
    """Structural check against guidance.schema.json. Returns error strings made of field paths only (never field values)."""
    errs = _obj(g, "$", TOP_REQUIRED, TOP_OPTIONAL)
    if not isinstance(g, dict):
        return errs
    ids = "guidance_id session_id host_id readout_id snapshot_last_entry_id".split()
    spec = {k: _ID for k in ids} | {"version": _UINT, "seq": _UINT}
    spec["schema_version"] = (_is_semver, "bad semver")
    spec["emitted_at"] = (_is_time, "bad date-time")
    spec["run_fingerprint"] = (_is_hex64, "bad sha256 hex")
    errs += _check(g, spec, "$") + _enum(g, "turn_kind", "$", TURN_KINDS)
    errs += _each(g.get("release", []), "$.release", _item)
    if g.get("turn_kind") == "user_turn":
        if "current_turn" not in g:
            errs.append("$.current_turn: required for a user_turn")
    else:
        errs += [f"$.{k}: only allowed for a user_turn" for k in TOP_OPTIONAL if k in g]
    if "current_turn" in g:
        errs += _turn(g["current_turn"], "$.current_turn")
    errs += _each(g.get("opened_tasks", []), "$.opened_tasks", _task)
    errs += _each(g.get("disposition_facts", []), "$.disposition_facts", _fact)
    ct = g.get("current_turn")
    if isinstance(ct, dict) and ct.get("H8") in ("refuse", "escalate"):
        if g.get("opened_tasks"):
            errs.append("$.opened_tasks: must be empty when H8 is refuse or escalate")
    elif "disposition_facts" in g:
        errs.append("$.disposition_facts: only allowed when H8 is refuse or escalate")
    return errs


def context_of(g) -> dict:
    """The part of a valid guidance that goes into generation: ids/versions and the structured facts, copied as given."""
    return {k: copy.deepcopy(g[k]) for k in CONTEXT_KEYS if k in g}


def message_of(ctx) -> str:
    return json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))


def accept(queue: list, payload, log=None) -> tuple[int, dict]:
    """Entry point behind /guide. Invalid -> (400, errors), logged. Otherwise queued."""
    log = log or logging.getLogger(__name__)
    errs = validate(payload)
    if errs:
        log.warning("sophon guide: rejected: %s", "; ".join(errs[:8]))
        return 400, {"ok": False, "errors": errs}
    gid, ver = payload["guidance_id"], payload["version"]
    queue.append(context_of(payload))
    log.info("sophon guide: queued %s v%d", gid, ver)
    return 200, {"ok": True, "guidance_id": gid, "version": ver}
