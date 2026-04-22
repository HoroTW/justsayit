#!/usr/bin/env python3
"""Evaluate how well a justsayit prompt aligns with its intended behavior.

Runs a curated TOML case suite through the real remote postprocess path
(``LLMPostprocessor.process_with_reasoning``), asks a judge LLM whether
each output was a ``cleanup`` or ``assistant`` response, and prints a
percentage score plus the list of failing cases. Intended as a tuning
loop for ``src/justsayit/prompts/cleanup_openai.md`` — NOT a pytest test
(real API calls, costs money, non-deterministic).

Typical use:
    uv run scripts/eval-cleanup-prompt.py --dry-run
    uv run scripts/eval-cleanup-prompt.py --yes
    uv run scripts/eval-cleanup-prompt.py --only bare-clipboard-question --runs 3 --yes

See evals/README.md for more.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The harness lives in scripts/ but imports from the project's src/
# layout. Add the package root so ``python scripts/eval-cleanup-prompt.py``
# works even without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from justsayit.config import resolve_secret  # noqa: E402
from justsayit.postprocess import (  # noqa: E402
    LLMPostprocessor,
    PostprocessProfile,
    load_profile,
)

SCHEMA_VERSION = 1
DEFAULT_CASES_PATH = _REPO_ROOT / "evals" / "cleanup_openai_cases.toml"
DEFAULT_PROFILE = "openai-cleanup"
MODE_CLEANUP = "cleanup"
MODE_ASSISTANT = "assistant"
MODE_UNPARSEABLE = "unparseable"
JUDGE_MAX_TOKENS = 300


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    description: str
    input: str
    clipboard: str = ""
    expected_mode: str = MODE_CLEANUP
    language: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class JudgeConfig:
    endpoint: str
    model: str
    api_key: str
    temperature: float
    request_timeout: float


@dataclass
class JudgeVerdict:
    mode: str
    reason: str
    raw: str


@dataclass
class RunOutcome:
    case: Case
    run_index: int
    model_output: str
    model_reasoning: str
    verdict: JudgeVerdict
    elapsed_ms: int

    @property
    def pass_(self) -> bool:
        return self.verdict.mode == self.case.expected_mode


@dataclass
class CaseResult:
    case: Case
    runs: list[RunOutcome] = field(default_factory=list)

    @property
    def majority_mode(self) -> str:
        ctr = Counter(r.verdict.mode for r in self.runs)
        # Ties break toward assistant — the user's main failure mode is
        # cleanup-expected being over-triggered; break toward the
        # "harder" failure so flakiness surfaces as a miss, not a pass.
        if not ctr:
            return MODE_UNPARSEABLE
        top = ctr.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            if MODE_ASSISTANT in ctr:
                return MODE_ASSISTANT
        return top[0][0]

    @property
    def flakiness(self) -> float:
        if not self.runs:
            return 0.0
        ctr = Counter(r.verdict.mode for r in self.runs)
        majority_count = ctr.most_common(1)[0][1]
        return 1.0 - majority_count / len(self.runs)

    @property
    def pass_(self) -> bool:
        return self.majority_mode == self.case.expected_mode


# ---------------------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    pass


def load_cases(path: Path) -> list[Case]:
    if not path.exists():
        raise ConfigError(f"cases file not found: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    schema = data.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ConfigError(
            f"cases file {path}: schema_version={schema!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    raw_cases = data.get("case", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigError(f"cases file {path}: no [[case]] entries found")
    seen_ids: set[str] = set()
    out: list[Case] = []
    for idx, raw in enumerate(raw_cases):
        prefix = f"case[{idx}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{prefix}: not a table")
        cid = raw.get("id")
        if not isinstance(cid, str) or not cid:
            raise ConfigError(f"{prefix}: missing/empty 'id'")
        prefix = f"case[{idx}] (id={cid})"
        if cid in seen_ids:
            raise ConfigError(f"{prefix}: duplicate id")
        seen_ids.add(cid)
        text = raw.get("input")
        if not isinstance(text, str) or not text:
            raise ConfigError(f"{prefix}: missing/empty 'input'")
        mode = raw.get("expected_mode")
        if mode not in (MODE_CLEANUP, MODE_ASSISTANT):
            raise ConfigError(
                f"{prefix}: expected_mode must be 'cleanup' or 'assistant', got {mode!r}"
            )
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ConfigError(f"{prefix}: 'tags' must be a list of strings")
        out.append(
            Case(
                id=cid,
                description=str(raw.get("description", "")),
                input=text,
                clipboard=str(raw.get("clipboard", "")),
                expected_mode=mode,
                language=str(raw.get("language", "")),
                tags=tuple(tags),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Target + judge setup
# ---------------------------------------------------------------------------


def build_target(profile_name_or_path: str) -> tuple[LLMPostprocessor, PostprocessProfile]:
    try:
        profile = load_profile(profile_name_or_path)
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from e
    if profile.base != "remote":
        raise ConfigError(
            f"profile {profile_name_or_path!r} has base={profile.base!r}; this "
            "harness only evaluates remote (OpenAI-compatible) profiles. Try "
            "--profile openai-cleanup or a user profile pointing at an endpoint."
        )
    if not profile.endpoint:
        raise ConfigError(
            f"profile {profile_name_or_path!r}: endpoint is empty"
        )
    if not profile.model:
        raise ConfigError(
            f"profile {profile_name_or_path!r}: model is empty"
        )
    key = resolve_secret(profile.api_key, profile.api_key_env)
    if not key:
        raise ConfigError(
            f"no API key found for profile {profile_name_or_path!r}: "
            f"set {profile.api_key_env!r} in the environment or in "
            f"~/.config/justsayit/.env"
        )
    # Install the resolved key so LLMPostprocessor doesn't re-resolve
    # (and so tests / --dry-run see exactly what would be used).
    profile.api_key = key
    post = LLMPostprocessor(profile)
    return post, profile


_OPENAI_MINI_RE = re.compile(r"mini|nano|small|4o-mini", re.I)


def resolve_judge_config(
    args: argparse.Namespace,
    target_profile: PostprocessProfile,
) -> JudgeConfig:
    endpoint = args.judge_endpoint or target_profile.endpoint
    if args.judge_model:
        model = args.judge_model
    elif _OPENAI_MINI_RE.search(target_profile.model or ""):
        model = "gpt-4o"
        print(
            f"note: judge-model defaulting to {model!r} because the target "
            f"({target_profile.model!r}) looks like a small model. "
            f"Override with --judge-model.",
            file=sys.stderr,
        )
    else:
        model = target_profile.model
        print(
            f"note: judge-model defaulting to target model {model!r}; "
            f"this introduces self-agreement bias. Override with --judge-model.",
            file=sys.stderr,
        )
    env_var = args.judge_api_key_env or target_profile.api_key_env
    key = resolve_secret("", env_var)
    if not key:
        raise ConfigError(
            f"no judge API key found: set {env_var!r} in the environment "
            f"or in ~/.config/justsayit/.env (override via --judge-api-key-env)"
        )
    return JudgeConfig(
        endpoint=endpoint,
        model=model,
        api_key=key,
        temperature=args.judge_temperature,
        request_timeout=float(target_profile.request_timeout or 60.0),
    )


# ---------------------------------------------------------------------------
# Target invocation (via production code path)
# ---------------------------------------------------------------------------


def run_target(post: LLMPostprocessor, case: Case) -> tuple[str, str]:
    result = post.process_with_reasoning(case.input, extra_context=case.clipboard)
    return result.text or "", result.reasoning or ""


def _base_prompt_text(profile: PostprocessProfile) -> str:
    """Return the base system prompt (minus dynamic/clipboard suffixes)
    so the report can track prompt length across edits."""
    if profile.system_prompt:
        return profile.system_prompt
    if profile.system_prompt_file:
        # Reuse the same resolution the runtime uses.
        from justsayit.postprocess import _resolve_system_prompt_file

        try:
            return _resolve_system_prompt_file(profile.system_prompt_file)
        except Exception:
            return ""
    return ""


def _approx_tokens(chars: int) -> int:
    # Rough rule of thumb: 4 chars ≈ 1 token for English/German mix.
    return chars // 4


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator for a dictation post-processing system. The system \
takes raw speech-to-text input and either (A) returns the transcript lightly \
cleaned — "cleanup mode" — or (B) answers a request / performs a task — \
"assistant mode".

Your ONLY job: given the raw input, the clipboard context (if any), and the \
system's actual output, classify the output into exactly one label:

  "cleanup"   — the output is functionally the input transcript (possibly \
with light cleanup edits) OR the system did not perform any meaningful \
transformation.
  "assistant" — the output is a substantive task response that is NOT a \
copy of either the input or the clipboard (answer, translation, rewrite, \
summary, explanation, or a hallucinated attempt at one).

Decision procedure (apply in order, stop at the first match):

1. **Echo of input (cleanup)** → "cleanup". Output equals the input, or \
is the input with ONLY light, non-content-bearing edits: filler removal \
("um", "uh", "ähm", "halt", "also", "like", "so"), spoken-punctuation → \
actual char, obvious STT mishear fix, dictated emoji substitution, minor \
capitalisation. Preserving the user's words — questions, greetings, modal \
particles, mixed languages — is cleanup. An empty output is cleanup.

   **NOT cleanup even if the output looks similar to the input**:
   - Output answers a question that was in the input.
   - Output drops an instruction token from the input ("translate", \
"summarise", "polish this note", "Hey Computer, …", "make this sound \
more formal") AND produces the result of executing that instruction.
   - Output substitutes words with ones that carry different meaning or \
tone ("talk" → "discuss", "I think" → "I believe", "work" → "function"). \
Filler removal does NOT count — those words carry no meaning.
   - Output is in a different language than the input (unless it is a \
faithful translation the user clearly requested; see clipboard rule below).

2. **Verbatim echo of clipboard** → "cleanup". If the clipboard is \
non-empty AND the output is the clipboard copied verbatim (or trivially \
close to it) with no translation / summary / rewrite applied, label \
"cleanup". The system received a task but performed no transformation — \
that is NOT a successful assistant response.

3. **Substantive task output** → "assistant". Otherwise, if the output \
answers a question, executes a task (translate / rewrite / summarise / \
polish / define / compute / look up), replies conversationally, \
translates into another language, OR produces unrelated hallucinated \
content, label "assistant". A botched-but-distinct attempt (wrong answer, \
off-topic response, unrelated sentence) is still "assistant" as long as \
the output is NOT a copy of the input or the clipboard.

Additional rules:
- Translation counts as "assistant" UNLESS the clipboard is non-empty and \
the input plausibly requests translation OF the clipboard content.
- Language mismatch between input and output (e.g. English input, German \
output) is "assistant" unless it is a faithful translation the user \
clearly requested.
- When the output is identical or near-identical to the input, it is \
always "cleanup" — even if the input was a question.
- Capitalisation, punctuation, and filler-removal differences alone do \
NOT push a response out of "cleanup".

Respond with JSON ONLY, no prose before or after, exactly this shape:

  {"mode": "cleanup" | "assistant", "reason": "<one sentence, <=30 words>"}
"""


def _judge_user_message(case: Case, model_output: str) -> str:
    return (
        "INPUT (raw dictation given to the system):\n"
        f"<<<\n{case.input}\n>>>\n\n"
        "CLIPBOARD CONTEXT (additional context the system received; may be empty):\n"
        f"<<<\n{case.clipboard}\n>>>\n\n"
        "SYSTEM OUTPUT (what the system actually produced):\n"
        f"<<<\n{model_output}\n>>>\n\n"
        "Classify the SYSTEM OUTPUT. Respond with JSON only."
    )


_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _call_judge_once(
    cfg: JudgeConfig,
    messages: list[dict[str, str]],
    want_json_format: bool,
) -> dict[str, Any]:
    url = cfg.endpoint.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
        "User-Agent": "justsayit-eval",
    }
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": JUDGE_MAX_TOKENS,
    }
    if want_json_format:
        body["response_format"] = {"type": "json_object"}
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return _post_json(url, headers, body, cfg.request_timeout)
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in _TRANSIENT_STATUS and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _extract_content(resp: dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.S)


def _parse_verdict(raw: str) -> JudgeVerdict | None:
    """Best-effort JSON extraction — some models wrap the JSON in prose
    despite the instructions. Return None on unrecoverable failure."""
    stripped = raw.strip()
    candidates: list[str] = [stripped]
    # Strip ```json fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", stripped, re.S)
    if fence:
        candidates.insert(0, fence.group(1))
    # Fall back to first {...} block.
    m = _JSON_BLOCK_RE.search(stripped)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            continue
        mode = parsed.get("mode") if isinstance(parsed, dict) else None
        if mode in (MODE_CLEANUP, MODE_ASSISTANT):
            reason = str(parsed.get("reason", "")).strip()
            return JudgeVerdict(mode=mode, reason=reason, raw=raw)
    return None


def run_judge(cfg: JudgeConfig, case: Case, model_output: str) -> JudgeVerdict:
    user_msg = _judge_user_message(case, model_output)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    want_json = cfg.endpoint.startswith("https://api.openai.com")
    try:
        resp = _call_judge_once(cfg, messages, want_json_format=want_json)
    except urllib.error.HTTPError as e:
        if e.code == 400 and want_json:
            # Some OpenAI-compatible servers 400 on response_format — retry without.
            resp = _call_judge_once(cfg, messages, want_json_format=False)
        else:
            raise
    content = _extract_content(resp)
    verdict = _parse_verdict(content)
    if verdict is not None:
        return verdict
    # One firmer retry: append a stricter JSON-only reminder.
    retry_messages = list(messages) + [
        {
            "role": "user",
            "content": (
                "Your previous response was not parseable JSON. "
                'Respond with ONLY the JSON object, e.g. '
                '{"mode":"cleanup","reason":"…"} — no prose, no fences.'
            ),
        }
    ]
    resp2 = _call_judge_once(cfg, retry_messages, want_json_format=want_json)
    content2 = _extract_content(resp2)
    verdict2 = _parse_verdict(content2)
    if verdict2 is not None:
        return verdict2
    return JudgeVerdict(mode=MODE_UNPARSEABLE, reason="judge response not parseable", raw=content2 or content)


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def evaluate(
    cases: list[Case],
    post: LLMPostprocessor,
    judge_cfg: JudgeConfig,
    runs: int,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    total = len(cases) * runs
    step = 0
    for case in cases:
        case_result = CaseResult(case=case)
        for r in range(runs):
            step += 1
            t0 = time.monotonic()
            try:
                model_output, model_reasoning = run_target(post, case)
            except Exception as e:
                model_output = f"<target call failed: {e!r}>"
                model_reasoning = ""
                verdict = JudgeVerdict(
                    mode=MODE_UNPARSEABLE,
                    reason=f"target call raised: {e!r}",
                    raw="",
                )
            else:
                try:
                    verdict = run_judge(judge_cfg, case, model_output)
                except Exception as e:
                    verdict = JudgeVerdict(
                        mode=MODE_UNPARSEABLE,
                        reason=f"judge call raised: {e!r}",
                        raw="",
                    )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            outcome = RunOutcome(
                case=case,
                run_index=r,
                model_output=model_output,
                model_reasoning=model_reasoning,
                verdict=verdict,
                elapsed_ms=elapsed_ms,
            )
            case_result.runs.append(outcome)
            tag = "PASS" if verdict.mode == case.expected_mode else "FAIL"
            print(
                f"[{step}/{total}] {case.id} run {r + 1}/{runs}: "
                f"{case.expected_mode}→{verdict.mode} {tag} ({elapsed_ms}ms)",
                file=sys.stderr,
            )
        results.append(case_result)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _confusion(results: list[CaseResult]) -> dict[tuple[str, str], int]:
    ctr: Counter[tuple[str, str]] = Counter()
    for cr in results:
        ctr[(cr.case.expected_mode, cr.majority_mode)] += 1
    return dict(ctr)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit} chars]"


def print_report(
    results: list[CaseResult],
    profile_name: str,
    target_profile: PostprocessProfile,
    judge_cfg: JudgeConfig,
    runs: int,
    elapsed_s: float,
    max_output_chars: int,
    show_passes: bool,
) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.pass_)
    cleanup_expected = [r for r in results if r.case.expected_mode == MODE_CLEANUP]
    assistant_expected = [r for r in results if r.case.expected_mode == MODE_ASSISTANT]
    cleanup_pass = sum(1 for r in cleanup_expected if r.pass_)
    assistant_pass = sum(1 for r in assistant_expected if r.pass_)
    target_calls = total * runs
    judge_calls = total * runs
    conf = _confusion(results)

    print()
    print("=== eval results ===")
    print(
        f"target : {profile_name}  ({target_profile.model} @ {target_profile.endpoint})"
    )
    print(
        f"judge  : {judge_cfg.model} @ {judge_cfg.endpoint}  (temperature={judge_cfg.temperature})"
    )
    print(
        f"cases  : {total}   runs-per-case: {runs}   "
        f"total calls: {target_calls} target + {judge_calls} judge = {target_calls + judge_calls}"
    )
    print(f"elapsed: {elapsed_s:.1f}s")
    # Base prompt size — report both raw chars and a cheap "efficiency"
    # metric. More-concise prompts that hold accuracy should visibly win.
    base_prompt = _base_prompt_text(target_profile)
    base_chars = len(base_prompt)
    print(f"prompt : {base_chars} chars ({_approx_tokens(base_chars)} approx tokens)")
    print()
    pct = (passed / total * 100.0) if total else 0.0
    eff = pct / (base_chars / 1000.0) if base_chars else 0.0
    print(f"overall: {passed}/{total} = {pct:.1f}%  (efficiency: {eff:.1f} %·1000chars⁻¹)")
    if cleanup_expected:
        c_pct = cleanup_pass / len(cleanup_expected) * 100.0
        print(
            f"  cleanup  (expected): {cleanup_pass}/{len(cleanup_expected)} = {c_pct:.1f}%"
        )
    if assistant_expected:
        a_pct = assistant_pass / len(assistant_expected) * 100.0
        print(
            f"  assistant(expected): {assistant_pass}/{len(assistant_expected)} = {a_pct:.1f}%"
        )
    print()
    print("confusion matrix (rows=expected, cols=judge label):")
    labels = [MODE_CLEANUP, MODE_ASSISTANT, MODE_UNPARSEABLE]
    header = "              " + "  ".join(f"{lab:>11}" for lab in labels)
    print(header)
    for expected in (MODE_CLEANUP, MODE_ASSISTANT):
        row = [conf.get((expected, lab), 0) for lab in labels]
        print(f"  {expected:<11}" + "  ".join(f"{n:>11}" for n in row))
    if runs > 1:
        flaky = [r for r in results if r.flakiness > 0.0]
        if flaky:
            print()
            print(
                f"flakiness: {len(flaky)}/{total} cases had non-unanimous votes "
                f"({len(flaky) / total * 100.0:.1f}%)"
            )

    def _print_block(label: str, cr: CaseResult) -> None:
        v = cr.runs[-1].verdict  # representative verdict (last run)
        print()
        print(
            f"--- {label}: {cr.case.id} "
            f"(expected {cr.case.expected_mode}, judged {cr.majority_mode}) ---"
        )
        if cr.case.description:
            print(f"description: {cr.case.description}")
        print(f"input      : {_truncate(cr.case.input, max_output_chars)}")
        clip = cr.case.clipboard or "(empty)"
        print(f"clipboard  : {_truncate(clip, max_output_chars)}")
        print(
            f"output     : {_truncate(cr.runs[-1].model_output or '(empty)', max_output_chars)}"
        )
        reasoning = cr.runs[-1].model_reasoning or "(none)"
        print(f"reasoning  : {_truncate(reasoning, max_output_chars)}")
        print(
            f"judge      : {v.mode} — \"{_truncate(v.reason, max_output_chars)}\""
        )
        if runs > 1 and cr.flakiness > 0.0:
            votes = Counter(r.verdict.mode for r in cr.runs)
            print(f"votes      : {dict(votes)} (flakiness={cr.flakiness:.2f})")

    failures = [r for r in results if not r.pass_]
    if failures:
        print()
        print(f"=== {len(failures)} failing case(s) ===")
        for cr in failures:
            _print_block("FAIL", cr)
    if show_passes:
        passes = [r for r in results if r.pass_]
        if passes:
            print()
            print(f"=== {len(passes)} passing case(s) ===")
            for cr in passes:
                _print_block("PASS", cr)


def write_json_report(
    results: list[CaseResult],
    profile_name: str,
    target_profile: PostprocessProfile,
    judge_cfg: JudgeConfig,
    runs: int,
    path: Path,
) -> None:
    payload: dict[str, Any] = {
        "target": {
            "profile": profile_name,
            "model": target_profile.model,
            "endpoint": target_profile.endpoint,
            "temperature": target_profile.temperature,
        },
        "judge": {
            "model": judge_cfg.model,
            "endpoint": judge_cfg.endpoint,
            "temperature": judge_cfg.temperature,
        },
        "runs_per_case": runs,
        "cases": [],
    }
    for cr in results:
        payload["cases"].append(
            {
                "id": cr.case.id,
                "description": cr.case.description,
                "input": cr.case.input,
                "clipboard": cr.case.clipboard,
                "expected_mode": cr.case.expected_mode,
                "language": cr.case.language,
                "tags": list(cr.case.tags),
                "majority_mode": cr.majority_mode,
                "flakiness": cr.flakiness,
                "pass": cr.pass_,
                "runs": [
                    {
                        "run_index": r.run_index,
                        "model_output": r.model_output,
                        "model_reasoning": r.model_reasoning,
                        "verdict_mode": r.verdict.mode,
                        "verdict_reason": r.verdict.reason,
                        "verdict_raw": r.verdict.raw,
                        "elapsed_ms": r.elapsed_ms,
                    }
                    for r in cr.runs
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _filter_cases(
    cases: list[Case], ids: list[str], tags: list[str]
) -> list[Case]:
    if ids:
        id_set = set(ids)
        cases = [c for c in cases if c.id in id_set]
        missing = id_set - {c.id for c in cases}
        if missing:
            raise ConfigError(
                "unknown case id(s): " + ", ".join(sorted(missing))
            )
    if tags:
        tag_set = set(tags)
        cases = [c for c in cases if tag_set.intersection(c.tags)]
    if not cases:
        raise ConfigError("no cases selected after filtering")
    return cases


def _confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="eval-cleanup-prompt.py",
        description=(
            "Run the justsayit cleanup/assistant-mode prompt against a suite "
            "of dictation cases and score the outputs via a judge LLM. "
            "Intended for iterating on src/justsayit/prompts/cleanup_openai.md "
            "— NOT a pytest test (real API calls)."
        ),
    )
    tgt = ap.add_argument_group("target model (the one you're evaluating)")
    tgt.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="justsayit postprocess profile (name or path). Default: %(default)s.",
    )
    tgt.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="TOML case file. Default: evals/cleanup_openai_cases.toml.",
    )
    jud = ap.add_argument_group("judge model (the one that labels outputs)")
    jud.add_argument(
        "--judge-endpoint",
        default=None,
        help="OpenAI-compatible base URL. Default: same as target profile.",
    )
    jud.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge model name. Default: 'gpt-4o' when target looks small "
            "(mini/nano/small), otherwise same as target."
        ),
    )
    jud.add_argument(
        "--judge-api-key-env",
        default=None,
        help="Env var holding the judge API key. Default: target profile's api_key_env.",
    )
    jud.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the judge. Default: %(default)s.",
    )
    run = ap.add_argument_group("run control")
    run.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repetitions per case. >1 → majority vote + flakiness metric. Default: %(default)s.",
    )
    run.add_argument(
        "--only",
        nargs="+",
        default=[],
        metavar="ID",
        help="Restrict to these case IDs.",
    )
    run.add_argument(
        "--tag",
        nargs="+",
        default=[],
        metavar="TAG",
        help="Restrict to cases with at least one of these tags.",
    )
    run.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write a machine-readable report to PATH.",
    )
    run.add_argument(
        "--show-passes",
        action="store_true",
        default=False,
        help="Include passes in the human report.",
    )
    run.add_argument(
        "--max-output-chars",
        type=int,
        default=400,
        help="Truncate each quoted string in the human report to N chars. Default: %(default)s.",
    )
    run.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip the cost-confirmation prompt.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Resolve profile + cases + judge config, print cost estimate, exit 0. No API calls.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.runs < 1:
        print("error: --runs must be >= 1", file=sys.stderr)
        return 1
    try:
        cases = load_cases(args.cases)
        cases = _filter_cases(cases, args.only, args.tag)
        post, profile = build_target(args.profile)
        judge_cfg = resolve_judge_config(args, profile)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    target_calls = len(cases) * args.runs
    judge_calls = len(cases) * args.runs
    print(
        f"target : {args.profile}  ({profile.model} @ {profile.endpoint})",
        file=sys.stderr,
    )
    print(
        f"judge  : {judge_cfg.model} @ {judge_cfg.endpoint}  "
        f"(temperature={judge_cfg.temperature})",
        file=sys.stderr,
    )
    print(
        f"cases  : {len(cases)}   runs-per-case: {args.runs}   "
        f"estimated calls: {target_calls} target + {judge_calls} judge = {target_calls + judge_calls}",
        file=sys.stderr,
    )
    if args.dry_run:
        print("dry-run: exiting without making API calls.", file=sys.stderr)
        return 0
    if not args.yes:
        print(
            "\nThese are real, paid API calls.",
            file=sys.stderr,
        )
        if not _confirm("Continue? [y/N] "):
            print("aborted.", file=sys.stderr)
            return 2

    t0 = time.monotonic()
    results = evaluate(cases, post, judge_cfg, args.runs)
    elapsed_s = time.monotonic() - t0

    print_report(
        results,
        profile_name=args.profile,
        target_profile=profile,
        judge_cfg=judge_cfg,
        runs=args.runs,
        elapsed_s=elapsed_s,
        max_output_chars=args.max_output_chars,
        show_passes=args.show_passes,
    )
    if args.json is not None:
        write_json_report(
            results,
            profile_name=args.profile,
            target_profile=profile,
            judge_cfg=judge_cfg,
            runs=args.runs,
            path=args.json,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
