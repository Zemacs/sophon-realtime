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

TOP_REQUIRED = ("schema_version", "guidance_id", "version", "session_id", "emitted_at", "readout_id", "snapshot_last_entry_id",
                "turn_kind", "release", "seq", "run_fingerprint", "host_id")
TOP_ALLOWED = TOP_REQUIRED + ("current_turn", "opened_tasks", "disposition_facts")
TURN_KINDS = ("user_turn", "event_turn", "timer_turn")
H1 = ("to_system", "feedback", "not_to_system", "holding")
H4 = ("immediate", "after_playback", "next_user_turn", "only_when_asked")
H7 = ("silent", "answer", "ask")
H8 = ("normal", "refuse", "escalate")
ITEM_KINDS = ("result", "progress", "failure", "playback_remainder", "reminder_due", "question")
NEEDS_EVIDENCE = ("result", "failure", "progress")
CONTEXT_KEYS = ("guidance_id", "version", "turn_kind", "current_turn", "opened_tasks", "release", "disposition_facts")

SYS_GUIDE = {"zh": "对话里 tool 角色的 JSON 是执行中枢给你的指导：只含结构化事实，不含现成的句子。请据此用自然口语说给用户，不要念字段名、编号或 JSON，也不要编造指导里没有的事实。",
             "en": "A tool-role JSON in the conversation is guidance from the execution hub: structured facts only, no ready-made sentences. "
                   "Say it to the user in natural spoken language; do not read out field names, ids or JSON, and do not invent facts that are not in it."}


def _is_id(v):
    return isinstance(v, str) and 1 <= len(v) <= 64 and v[0] in _ALNUM and all(c in _ID_REST for c in v)


def _is_uint(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _is_semver(v):
    parts = v.split(".") if isinstance(v, str) else []
    return len(parts) == 3 and all(p and all(c in _DIGITS for c in p) for p in parts)


def _is_hex64(v):
    return isinstance(v, str) and len(v) == 64 and all(c in _HEX for c in v)


def _is_time(v):
    if not isinstance(v, str):
        return False
    try:
        return datetime.fromisoformat(v[:-1] + "+00:00" if v.endswith("Z") else v).tzinfo is not None
    except ValueError:
        return False


def _obj(v, path, required, allowed, errs):
    if not isinstance(v, dict):
        errs.append(f"{path}: must be an object")
        return False
    errs.extend(f"{path}.{k}: required" for k in required if k not in v)
    errs.extend(f"{path}.{k}: not allowed" for k in v if k not in allowed)
    return True


def _enum(v, path, options, errs):
    if not (isinstance(v, str) and v in options):
        errs.append(f"{path}: must be one of {'/'.join(options)}")


def _array(v, path, errs):
    if not isinstance(v, list):
        errs.append(f"{path}: must be an array")
        return []
    return v


def _fact(f, path, errs):
    if not _obj(f, path, ("key", "value"), ("key", "value", "unit"), errs):
        return
    if "key" in f and not _is_id(f["key"]):
        errs.append(f"{path}.key: bad id")
    v = f.get("value")
    if "value" in f and not (v is None or isinstance(v, (int, float)) or (isinstance(v, str) and len(v) <= 64)):
        errs.append(f"{path}.value: must be a scalar (string at most 64 chars, number, boolean or null)")
    if "unit" in f and not _is_id(f["unit"]):
        errs.append(f"{path}.unit: bad id")


def _item(it, path, errs):
    if not _obj(it, path, ("item_id", "item_version", "kind", "released_by", "facts"),
                ("item_id", "item_version", "kind", "released_by", "facts", "evidence"), errs):
        return
    if "item_id" in it and not _is_id(it["item_id"]):
        errs.append(f"{path}.item_id: bad id")
    if "item_version" in it and not _is_uint(it["item_version"]):
        errs.append(f"{path}.item_version: must be a non-negative integer")
    if "kind" in it:
        _enum(it["kind"], f"{path}.kind", ITEM_KINDS, errs)
    if "released_by" in it:
        _enum(it["released_by"], f"{path}.released_by", H4, errs)
    for i, f in enumerate(_array(it.get("facts", []), f"{path}.facts", errs)):
        _fact(f, f"{path}.facts[{i}]", errs)
    if "evidence" in it or it.get("kind") in NEEDS_EVIDENCE:
        ev = _array(it.get("evidence"), f"{path}.evidence", errs) if "evidence" in it else []
        if it.get("kind") in NEEDS_EVIDENCE and not ev:
            errs.append(f"{path}.evidence: at least one reference is required for this kind")
        for i, r in enumerate(ev):
            if _obj(r, f"{path}.evidence[{i}]", ("id", "version"), ("id", "version"), errs):
                if "id" in r and not _is_id(r["id"]):
                    errs.append(f"{path}.evidence[{i}].id: bad id")
                if "version" in r and not _is_uint(r["version"]):
                    errs.append(f"{path}.evidence[{i}].version: must be a non-negative integer")


def validate(g) -> list[str]:
    """Structural check against guidance.schema.json. Returns error strings made of field paths only (never field values)."""
    errs: list[str] = []
    if not _obj(g, "$", TOP_REQUIRED, TOP_ALLOWED, errs):
        return errs
    formats = {"schema_version": _is_semver, "guidance_id": _is_id, "version": _is_uint, "session_id": _is_id, "emitted_at": _is_time, "seq": _is_uint,
               "run_fingerprint": _is_hex64, "host_id": _is_id, "readout_id": _is_id, "snapshot_last_entry_id": _is_id}
    errs.extend(f"$.{k}: bad format" for k, ok in formats.items() if k in g and not ok(g[k]))
    if "turn_kind" in g:
        _enum(g["turn_kind"], "$.turn_kind", TURN_KINDS, errs)
    for i, it in enumerate(_array(g.get("release", []), "$.release", errs)):
        _item(it, f"$.release[{i}]", errs)
    user = g.get("turn_kind") == "user_turn"
    if user and "current_turn" not in g:
        errs.append("$.current_turn: required for a user_turn")
    if not user:
        errs.extend(f"$.{k}: only allowed for a user_turn" for k in ("current_turn", "opened_tasks", "disposition_facts") if k in g)
    ct = g.get("current_turn")
    if "current_turn" in g and _obj(ct, "$.current_turn", ("H1", "H7", "H8"), ("H1", "H7", "H8"), errs):
        for k, options in (("H1", H1), ("H7", H7), ("H8", H8)):
            if k in ct:
                _enum(ct[k], f"$.current_turn.{k}", options, errs)
    for i, t in enumerate(_array(g.get("opened_tasks", []), "$.opened_tasks", errs)):
        if _obj(t, f"$.opened_tasks[{i}]", ("task_id", "task_version"), ("task_id", "task_version", "acceptance_event_id"), errs):
            errs.extend(f"$.opened_tasks[{i}].{k}: bad id" for k in ("task_id", "acceptance_event_id") if k in t and not _is_id(t[k]))
            if "task_version" in t and not _is_uint(t["task_version"]):
                errs.append(f"$.opened_tasks[{i}].task_version: must be a non-negative integer")
    for i, f in enumerate(_array(g.get("disposition_facts", []), "$.disposition_facts", errs)):
        _fact(f, f"$.disposition_facts[{i}]", errs)
    declined = isinstance(ct, dict) and ct.get("H8") in ("refuse", "escalate")
    if declined and g.get("opened_tasks"):
        errs.append("$.opened_tasks: must be empty when H8 is refuse or escalate")
    if not declined and "disposition_facts" in g:
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
        log.warning("sophon guide: rejected, %d error(s): %s", len(errs), "; ".join(errs[:8]))
        return 400, {"ok": False, "errors": errs}
    gid, ver = payload["guidance_id"], payload["version"]
    queue.append(context_of(payload))
    log.info("sophon guide: queued %s v%d (%d release item(s))", gid, ver, len(payload["release"]))
    return 200, {"ok": True, "guidance_id": gid, "version": ver}
