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
| gpt-5.4-nano (reasoning medium) | same | 94.7% | 12/12 | 6/7 (flaky) |
| gpt-5.4-mini (reasoning medium) | same | **100.0%** | 12/12 | 7/7 |

Latency comparison — target-only, 15 runs each across 5 representative cases:

| target | median | mean | p90 | max |
|---|---|---|---|---|
| gpt-4o-mini | 754 ms | 782 ms | 1259 ms | 1337 ms |
| gpt-5.4-mini (reasoning medium) | 1033 ms | 1068 ms | 1516 ms | 1601 ms |
| gpt-5.4-nano (reasoning medium) | 1204 ms | 1276 ms | 1843 ms | 1851 ms |
| gpt-5-mini (reasoning medium) | 3912 ms | 4984 ms | 7977 ms | 10320 ms |

`gpt-5.4-nano` target-only probe showed `clipboard-translate-de` is
2/3 correct, 1/3 echoed — single-run eval rolls the failure side
sometimes. Honest nano score is between 94.7% and 100% depending on
sampling; use `--runs 3+` when comparing nano vs mini more carefully.

`gpt-5-mini` is the **slowest** of the reasoning options (3-10s
vs 0.7-1.6s for the gpt-5.4 siblings) and scores 94.7% with a
different failure mode: it treats reported/quoted `hey computer`
as a real trigger and responds conversationally, instead of echoing.
Strictly worse on latency AND accuracy compared to gpt-5.4-mini.

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
