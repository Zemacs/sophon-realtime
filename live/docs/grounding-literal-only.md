# Round B (G-1): the grounding literal path is an exact comparison

Critic's b11afd4 review (G-1) found that the literal ("was this value said?") path of `grounding.check` still made judgments of meaning with tables:
English number words, currency words, months / ordinals, "twenty twenty six", a rebuilt spoken e-mail, grouped number words (`word_numbers`,
`spoken_forms`, `joins_n`), three lenient rules (a date whose year was never said; an e-mail matched up to its first domain label; a multi-word value whose
parts occur anywhere) and a short-value exemption.

## What is left on the literal path
`norm_text` (NFKC, case, drop everything but word characters / `@` / `.`), `occurs` (substring with a digit boundary: 88 is not in 1988), `_turns_n`
(turns joined with `|`), and the whole-word test for a Latin token (`appears_in_words`: Daniel is not Danielle). Nothing else.

## 行为变化 (behaviour changes)
| Before | Now |
|---|---|
| "four thousand" / "two hundred fifty dollars" grounded 4000 / 250 | not literal. The model must attach `grounding_evidence` (span) and an independent JEV judge must say yes |
| "we won" grounded `krw`; "for real" grounded `brl`; "twenty bucks" grounded `usd` | not literal (same path) |
| "december 14th" grounded `1999-12-14` (year never said) | not literal |
| "jon at acme dot com" grounded `jon@acme.com`; `john.smith@acme.org` grounded by "john.smith at acme" | not literal |
| "one nine one two two" grounded 19122; "Y, U, S, U, F" grounded `yusuf` | not literal (`1 9 1 2 2` is still the same text after `norm_text` drops spaces) |
| "Wang Li Si" grounded when Wang, Li, Si each occurred anywhere | not literal |
| values under 3 characters / one CJK character were never checked (`TX`, `京`) | checked like every other value (only a value that is empty after normalisation, i.e. punctuation only, is skipped) |

Fail closed: judge no, `None`, an exception, a timeout, a non-bool answer or no judge = not grounded; the model then reads the fixed `guard_message`
("NOT EXECUTED ... a second model checks that the span says the value"). The spell-it-out hint (`spelled_joins`) is display only and no longer reads number words.

## Latency
Every spoken form that used to pass on the literal path (zero model calls) now costs one more JEV judgment (`span_judge`), made before the tool is released
(`realtime._guard` awaits it): up to the 1.5 s `model_client` JEV budget (+ one retry on connection errors). Refusals while the model has no evidence yet
add a model turn (the refusal asks it to list the span). Numbers relayed from Critic's review (~0.4-0.5 s typical); not re-measured (no server access).

## Tests
`tests/test_r4m_g1_literal_only.py` (20 tests: Critic's probe rows, the same values released only by evidence + yes, fail-closed variants through
`check_async`, no short-value exemption, what stays literal, removed machinery stays removed), `tests/test_no_semantic_rules.py` (the grounding module may only
hold allowlisted closed sets / format / whole-token patterns; planted number / month tables and a spoken-e-mail regex are new hits), and 14 new mutants
in `tests/mutation_check_r4c.py` (`G-1: ...`). Old tests that asserted the removed rules were rewritten to the new expectation (spoken forms are refused alone and
released by evidence + yes); none was deleted.
