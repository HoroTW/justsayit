# Prompt evaluation

Tooling for iterating on `src/justsayit/prompts/cleanup_openai.md` (and
eventually the other prompts). Runs a curated TOML case suite through
the real remote postprocess path, asks a judge LLM whether each output
was a `cleanup` or `assistant` response, and reports a percentage plus
the failing cases so the prompt can be tightened.

This is a developer tool, not a regression suite — it makes real API
calls and costs real money. It is deliberately not a pytest test and
not a `justsayit` subcommand.

## Files

- `cleanup_openai_cases.toml` — the case suite (dictation inputs +
  expected modes). Edit this to add cases; do **not** copy failing
  inputs into `cleanup_openai.md` as examples — generalize the rule
  instead.
- `../scripts/eval-cleanup-prompt.py` — the runner.

## Quick start

```sh
# 1. Verify wiring without spending anything.
uv run scripts/eval-cleanup-prompt.py --dry-run

# 2. Smoke test on one cheap case.
uv run scripts/eval-cleanup-prompt.py --only very-short-cleanup --yes

# 3. Full run.
uv run scripts/eval-cleanup-prompt.py --yes

# 4. Iterate on a single failing case.
uv run scripts/eval-cleanup-prompt.py --only bare-clipboard-question --runs 3 --yes
```

The script reads `OPENAI_API_KEY` from the process env or from
`~/.config/justsayit/.env` (same resolution as the running app, via
`justsayit.config.resolve_secret`).

## Reading the report

```
overall: 15/18 = 83.3%
  cleanup  (expected): 12/14 = 85.7%
  assistant(expected):  3/4  = 75.0%

confusion matrix (rows=expected, cols=judge label):
              cleanup  assistant  unparseable
  cleanup         12          2            0
  assistant        1          3            0
```

- **`cleanup → assistant` failures** (top-right) mean the prompt is too
  eager to invoke assistant mode on bare dictation. This is the user's
  original bug. Tighten the "HARD REQUIREMENT: literal `Computer` must
  be present" section or strengthen the "never translate unless asked"
  rule.
- **`assistant → cleanup` failures** (bottom-left) mean the trigger
  isn't being respected. Usually a new phrasing variant needs
  coverage.
- **`unparseable`** means the judge didn't return JSON. Rare; retry
  once is already built in. If it recurs, add the raw text to a bug
  report.

## Flags worth knowing

- `--runs N` repeats each case N times; the per-case label becomes the
  majority vote and a flakiness metric is printed. Use after a
  single-run score looks good, to confirm improvements aren't noise
  (one case flipping = 5.5 percentage points with 18 cases).
- `--only ID [ID ...]` and `--tag TAG [TAG ...]` restrict the run
  while iterating.
- `--json PATH` dumps full (untruncated) per-run data for diffing two
  eval runs.
- `--show-passes` includes passing cases in the human report.
- `--dry-run` resolves config and prints the cost estimate without
  calling any API.

## Judge-bias note

Default: judge uses the same provider as the target (typically OpenAI
`gpt-4o` judging OpenAI `gpt-4o-mini`). That's friction-free but
introduces self-agreement bias.

For a more independent judge, point at OpenRouter (their API is
OpenAI-compatible, unlike Anthropic's native API):

```sh
export OPENROUTER_API_KEY=sk-or-…
uv run scripts/eval-cleanup-prompt.py \
    --judge-endpoint https://openrouter.ai/api/v1 \
    --judge-model anthropic/claude-sonnet-4.6 \
    --judge-api-key-env OPENROUTER_API_KEY \
    --yes
```

## Adding cases

Each `[[case]]` block in `cleanup_openai_cases.toml` needs:

- `id` — stable short slug (used in reports and `--only`).
- `description` — one line; why this case is here.
- `input` — the raw dictation.
- `expected_mode` — `"cleanup"` or `"assistant"`.

Optional:

- `clipboard` — when non-empty, passed as `extra_context` to the
  postprocessor. The system prompt currently forces assistant mode
  whenever a clipboard is attached (`postprocess.py:645-653`), so a
  clipboard-present case should almost always be `expected_mode =
  "assistant"`.
- `language` — free-form tag (`"de"`, `"en"`, `"mix"`, …).
- `tags` — list of tags for `--tag` filtering.

Keep the suite small and representative. Large suites slow iteration
and raise cost without adding coverage.

## Cost estimate

18 cases × 1 run × (1 target call + 1 judge call) = 36 API calls per
full run. At `gpt-4o-mini` target + `gpt-4o` judge this is typically
5–15 ¢ per run. `--runs 3` triples it. Set a spending limit on the
provider side if you plan to iterate a lot.
