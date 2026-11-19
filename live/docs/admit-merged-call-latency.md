# V-1: the merged `admit` JEV call (route + control + tool in one request)

Round A keeps rc3's structure: `coordinator.api._admit` sends ONE request to the router with the `route`, `control` and `tool` questions
(`task_faithful` is excluded). Before, the tool decision was an in-process regex (0 requests, microseconds); the route request carried only
`route` + `control`.

| | before (r4) | now |
|---|---|---|
| JEV requests per admitted turn | 1 (`route` + `control`) | 1 (`route` + `control` + `tool`): a longer prompt (the tool question's criteria) and one more answer decoded |
| Extra requests | none | CLARIFY route: +1 (`missing_slot`, rare) |
| JEV p50 *(relayed from Critic V-1, **not re-measured here**: no server access)* | 156 ms | 433 ms |
| Budget / failure | `model_client.budget("jev")` 1.5 s; failure -> keyword fallback | same budget; failure -> `_degrade` (model-free, no keywords) |

This sits on the path before the filler / lead-in, so every turn pays about +280 ms *(relayed)*.

Alternative, not taken in this round: a separate small `tool` request issued only when the route is WORKER or CONTROL->NEW, so FAST and pure control
turns do not pay for it, at the cost of one more request on tool turns (relayed: a parallel two-call experiment was dropped because JEV is
single-stream, 582 ms vs 512 ms serial). To be measured on the live server once the user lifts the freeze; the decision is then data, not opinion.

The invalid / missing tool answers are counted (`/health?detail=1` -> `router_invalid_answers`), so a prompt that is too long for the model
shows up as a rate, not as silence.
