# Improving the cleanup prompt — workflow notes

Pick-up doc for future iteration sessions. If the previous pass left
`cleanup_openai.md` in a decent state and you (or a new Claude session)
want to push it further, read this first.

## The loop

```
(1) collect failing cases  →  (2) target-only inspect  →  (3) edit prompt
        ↑                                                        │
        └──────── (5) full judge eval ──── (4) re-inspect ───────┘
```

1. **Collect failing cases.** Either from real use ("this dictation came
   out wrong") or from `evals/cleanup_openai_cases.toml`. Each case is
   `(input, clipboard?, expected_mode, expected_output)`. Add new real
   failures as TOML entries — they stay in the regression suite forever.

2. **Target-only inspect first.** Always. Budget-wise the judge is the
   expensive part; iterating against raw target outputs is close to
   free and is how you catch everything that matters:

   ```sh
   uv run scripts/eval-cleanup-prompt.py --no-judge --yes \
       --runs 5 --only case-id-1 case-id-2
   ```

   `--runs 5` surfaces determinism. A case that's `3/5 right, 2/5
   wrong` is a DIFFERENT problem from `5/5 wrong` — the first is a
   temperature/sampling issue, the second is a prompt or model issue.

3. **Edit the prompt.** Minimum viable change — one rule, one example,
   one reordering. DO NOT stack multiple edits before re-running.
   Prompts behave unpredictably; isolating the change is the only
   way to know what moved the needle.

4. **Re-inspect with `--no-judge`.** Confirm the output changed. If
   the failing case's output didn't change: the model didn't "see"
   your edit — try a stronger / different location for the rule.

5. **Only when target-only outputs look right: run the full judge
   eval.** Usually 1 full run per meaningful prompt state, at the
   end. Use `--json evals/runs/<descriptive-name>.json` so runs
   are diff-able later.

## Landmines this project has already walked onto

These are specific enough that you will waste time rediscovering them
if you don't read this list first.

- **Lenient judge inflation.** A judge that says "near-identical to
  input → cleanup" will quietly classify `Computer, translate this
  to German: hello world` → `hello world` as cleanup. That's the
  model partially executing the instruction, which is assistant
  mode. The current judge rubric in `scripts/eval-cleanup-prompt.py`
  explicitly rejects "partial extraction" as cleanup. Don't soften it.
- **Prompt token leakage.** A small model will output a capitalised
  label it saw in the prompt as a literal token (`CLEANUP only`,
  `CLEANUP mode is active…`). Avoid bare `CLEANUP` in the prompt;
  `cleanup` / `echo` / "stay in cleanup" are safer.
- **Failing-input echo bans.** Project memory says: when the prompt
  misbehaves on input X, add a *general rule*, not X as an example.
  Copying the failing dictation verbatim into the prompt as a
  counter-example bloats the prompt and patches one phrase.
- **Assistant-mode over-trigger.** gpt-4o-mini cannot reliably
  resist `translate this to German: …` patterns when they appear
  in the input, regardless of prompt wording. This is a model
  capability ceiling, not a prompt bug. Upgrade to a reasoning
  model (gpt-5.4-mini with `reasoning_effort="medium"` clears it
  100% — see `~/.config/justsayit/postprocess/gpt-5.4-mini.toml`).
- **Self-agreement bias.** Default judge is OpenAI (matches the
  target provider). For claims about absolute quality, cross-check
  once via OpenRouter with a Claude judge — recipe in
  `evals/README.md`.
- **`response_format` portability.** Gated to
  `endpoint.startswith("https://api.openai.com")` in the harness —
  other OpenAI-compatible servers 400 on it.
- **Reasoning models reject classic sampling knobs.** `temperature`
  must be 1, `top_p` / presence / frequency penalties are rejected,
  `max_tokens` → `max_completion_tokens`. `postprocess.py:_remote_process`
  detects by model name (`o[1-9]` / `gpt-[5-9]`) and strips them.
- **Clipboard present = always assistant.** When
  `# Clipboard as additional context` is attached the system prompt
  forces assistant mode. A case with `clipboard = "..."` and
  `expected_mode = "cleanup"` is structurally impossible to
  construct today — don't write one.
- **Small N uncertainty.** 19 cases at single-run → one flip =
  5.3 pp. Treat differences below ~5 pp as noise; `--runs 3` plus
  flakiness metric helps.

## Result history (as of latest commit)

| target | prompt | honest overall | cleanup | assistant |
|---|---|---|---|---|
| gpt-4o-mini | current `Hey`-only | 84.2% | 10/12 | 6/7 |
| gpt-5-mini (reasoning medium) | same | 94.7% | 11/12 | 7/7 |
| gpt-5.4-nano (reasoning low) | same | 78.9% | 12/12 | 3/7 |
| gpt-5.4-nano (reasoning low, *nano-tuned*) | cleanup_openai_nano.md | 89.5% | 12/12 | 5/7 |
| gpt-5.4-nano (reasoning medium) | same | 94.7% | 12/12 | 6/7 (flaky) |
| gpt-5.4-nano (reasoning medium, *nano-tuned*) | cleanup_openai_nano.md | 94.7% | 12/12 | 6/7 |
| gpt-5.4-nano (reasoning high, *nano-tuned*) | cleanup_openai_nano.md | **100.0%** | 12/12 | 7/7 |
| gpt-5.4-mini (reasoning low) | same | **100.0%** | 12/12 | 7/7 |
| gpt-5.4-mini (reasoning medium) | same | 100.0% | 12/12 | 7/7 |

Latency comparison — target-only, 15 runs each across 5 representative cases:

| target | median | mean | p90 | max |
|---|---|---|---|---|
| gpt-4o-mini | 754 ms | 782 ms | 1259 ms | 1337 ms |
| **gpt-5.4-mini (reasoning low)** | **781 ms** | 792 ms | **992 ms** | 1086 ms |
| gpt-5.4-mini (reasoning medium) | 1033 ms | 1068 ms | 1516 ms | 1601 ms |
| gpt-5.4-nano (reasoning low) | 900 ms | 936 ms | 1179 ms | 1278 ms |
| gpt-5.4-nano (reasoning medium) | 1204 ms | 1276 ms | 1843 ms | 1851 ms |
| gpt-5.4-nano (reasoning high, nano-tuned) | 1153 ms | 1417 ms | 2378 ms | 2464 ms |
| gpt-5-mini (reasoning medium) | 3912 ms | 4984 ms | 7977 ms | 10320 ms |

## Winners (pick one)

- **gpt-5.4-mini @ reasoning=low** with the default `cleanup_openai.md`.
  100% accuracy, 781 ms median, 992 ms p90 (lower p90 than gpt-4o-mini
  because the reasoning model is more consistent). Profile lives at
  `~/.config/justsayit/postprocess/gpt-5.4-mini-low.toml`. This is
  the recommended default — approaches gpt-4o-mini speed at full
  correctness.
- **gpt-5.4-nano @ reasoning=high** with the `cleanup_openai_nano.md`
  variant. Also 100% accuracy but p90 is 2.4x the mini-low p90, so
  there is no speed or p99-consistency reason to prefer it. If nano's
  per-token cost at `reasoning=high` ends up meaningfully cheaper
  than mini@low in practice, worth a look — otherwise not.

## Lessons learned in this tuning round

- `reasoning=low` is enough for mini to maintain 100%. The `medium`
  default is leaving latency on the table; drop to `low`.
- `reasoning=low` is NOT enough for nano — it becomes too conservative
  and echoes `Hey Computer, …` inputs instead of acting. Nano
  apparently uses the reasoning pass to commit to assistant mode.
- A nano-specific prompt tweak (stronger "when triggered you MUST
  act; echoing is failure" reminder at the end of the prompt, saved
  as `cleanup_openai_nano.md`) raised nano@low from 78.9% to 89.5%
  but didn't close the gap to mini@low's 100%. The last 10% requires
  more reasoning budget.
- Per-model prompts are fine when they help. Keep the default
  `cleanup_openai.md` unchanged for the mini/4o-mini profiles;
  point the nano profile at `cleanup_openai_nano.md`.

## Kick-off prompt for a new iteration

Paste this verbatim to Claude when you want to start a fresh round.
Add any new failing cases you've observed in real use at the bottom.

> I want to do another improvement round on `cleanup_openai.md`. Read
> `evals/IMPROVING.md` first — it has the workflow, the landmines
> we've already hit, and the kick-off expectations. Then:
>
> 1. Run `scripts/eval-cleanup-prompt.py --yes` (gpt-4o-mini +
>    current prompt) to get the current honest score as a baseline.
> 2. For every failing case, run `--no-judge --runs 5 --only <id>`
>    to check determinism before proposing a prompt change. Target-
>    only inspection is cheap; judge runs are not.
> 3. Make the smallest prompt edit that could plausibly fix the
>    determinism-confirmed failures. One edit at a time.
> 4. Re-inspect with `--no-judge` before running the full judge
>    eval again.
> 5. Commit only when the full-suite honest score improved or
>    when a deliberate trade-off is explicitly called out in the
>    commit message.
>
> New failing cases I want to add to `evals/cleanup_openai_cases.toml`
> (dictation observed in real use that didn't cleanup correctly):
>
> - id: `<slug>` — input: `<text>` — clipboard: `<text or empty>` —
>   expected_mode: `<cleanup|assistant>` — expected_output (optional):
>   `<text>` — notes: `<what the model actually did>`
> - …
>
> Budget: up to 5 full judge evals (~95 judge calls) and up to ~300
> target calls. Use `--no-judge` liberally; it's free relative to
> that ceiling.

## What lives where

- `evals/cleanup_openai_cases.toml` — the test suite. Add new real
  failures here; never delete cases, they're regressions.
- `scripts/eval-cleanup-prompt.py` — the harness. Judge rubric lives
  in `JUDGE_SYSTEM_PROMPT`; tighten if you catch a false-pass.
- `evals/runs/*.json` — full per-run results, kept for diffing. Name
  them descriptively (`baseline-v3.json`, `variant-a.json`,
  `final.json`, `gpt-5.4-mini-reasoning-medium.json`, etc.).
- `src/justsayit/prompts/cleanup_openai.md` — the prompt under test.
- `~/.config/justsayit/postprocess/gpt-5.4-mini.toml` — local-only
  profile for A/B comparison against gpt-4o-mini. Run
  `--profile gpt-5.4-mini` to eval with reasoning-capable target.
