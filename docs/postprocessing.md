# LLM postprocessing

An optional LLM cleanup pass runs after transcription. The default
"cleanup" profile fixes obvious mishears, removes filler words, applies
dictated formatting / punctuation, and can switch into [assistant mode
when `Hey Computer` appears as an actual cue to the model](#hey-computer--inline-assistant-mode).
The same machinery can do anything else you'd ask an LLM to do —
emojify, translate, summarise, change tone, format as Markdown, etc. —
by swapping in a custom system prompt.

Enable in `config.toml` (or toggle from the tray):

```toml
[postprocess]
enabled = true
profile = "gemma4-cleanup"   # filename stem under postprocess/
dynamic_context_script = "~/.config/justsayit/dynamic-context.sh"
```

`justsayit init` writes the default `dynamic-context.sh` helper if it is
missing. justsayit runs that bash script on every LLM request with a
small timeout. When stdout is non-empty, it is prepended before the
normal system prompt exactly like this:

```text
# STATE (DYNAMIC CONTEXT):
$RESULT

----

$SYSTEM_PROMPT
```

If the script prints nothing, the state block is omitted entirely. The
shipped script uses only local, no-network heuristics and prints local
time, date, timezone, and a locale hint when available.

Set `postprocess.dynamic_context_script = ""` to disable dynamic
context entirely.

## "Hey Computer" — inline assistant mode

The shipped cleanup profiles double as a zero-friction assistant.
`Hey Computer` anywhere in the transcript is generally treated as a cue
that the text is meant for the model, so the reply lands in your focused
window the same way a transcription would.

- **Leading trigger stays the main rule.** A leading `Hey Computer`
  (case-insensitive) is still the clearest and most reliable pattern.
  Common STT mishears like `Hi Computer` / `Hey Computa` are tolerated.
- **Not a hard parser rule.** The shipped prompts are intentionally
  broader than "must be at the start": if `Hey Computer` appears later
  and it plainly reads like a request to the model, the model may treat
  it as assistant mode too. This is best-effort prompt behavior, not a
  deterministic app-side mode switch.
- **Quoted / reported / incidental uses stay cleanup-only.** If the text
  is clearly reporting speech, quoting someone else, naming something,
  or otherwise not actually addressing the model, it should remain plain
  cleanup. The same applies when treating it as an instruction clearly
  does not make sense.
- **No mode-switch UI.** The trigger lives inside the system prompt;
  same hotkey, same overlay, same paste flow.
- **Best effort, not a guarantee.** README/docs describe the intended
  semantics of the shipped prompt. Different models may still vary at
  the edges, especially on ambiguous transcripts.
- **Direct replies.** When triggered, the model answers without
  echoing your request and without preamble like "Sure, here you go:".

Examples:

| You say | Result |
|---------|--------|
| `Hey Computer, what's 47 times 18?` | `846` |
| `hey computer translate to German: see you tomorrow` | `Bis morgen` |
| `Hey Computer, make this sound more formal` | (rewritten in a more formal tone) |
| `Hey Computer, there is an offering, please write a humble decline with the wording 'deeply sorry ...'` | (a short humble decline using that wording) |
| `Hey Computer, write a Python one-liner that sums a list of dicts by key 'cost'.` | `sum(d['cost'] for d in items)` |
| `Hey Computer, give me a polite decline for a meeting on Friday.` | (a short polite decline) |
| `this is my rough follow-up note hey computer please clean this up` | (best-effort assistant-style rewrite of the earlier dictated text) |
| `please polish this note for the client hey computer make this sound more formal` | (best-effort assistant-style rewrite of the earlier dictated text) |
| `Computer, translate this to German: hello world` | (cleanup only — bare `Computer` is **not** the trigger) |
| `Can you tell me how many things you can see?` | (cleanup only — no trigger) |
| `and then I told him hey computer remind me tomorrow` | (cleanup only — quoted/reported, not actually addressed to the model) |
| `the folder is called hey computer drafts` | (cleanup only — incidental mention, not an instruction) |
| `this is my rough note hey computer translate to German` | (often assistant-mode best effort, unless the transcript clearly reads as something else) |

The trigger behavior lives in the system prompt. If you want a
different wake word, a different language, or stricter / looser trigger
semantics, edit your profile's `system_prompt` and the rules change with
it. Custom profiles (translate, emojify, summarise, …) typically drop the
assistant-mode block entirely so they always rewrite, never branch.

## Emoji phrases

The shipped cleanup prompts now make the emoji rule explicit: when a
spoken emoji phrase is clearly intended, the whole dictated phrase should
collapse to just the emoji, including slight STT mishears. Example:
`Fragen da Emoji` should become `🤔`, not `Fragen da 🤔`.

## Shipped profiles

`justsayit init` writes three profiles into `~/.config/justsayit/postprocess/`:

| Profile | Backend | What it does |
|---------|---------|--------------|
| `gemma4-cleanup` | Local Gemma 4 E4B via `llama-cpp-python` | Recommended. Conservative DE/EN cleanup tuned for Gemma. Treats `Hey Computer` as a best-effort assistant cue anywhere in the transcript, while quoted/reported/incidental uses should remain cleanup-only. |
| `gemma4-fun` | Same local Gemma model | Keeps your wording but sprinkles emojis. Great for chat / social. |
| `openai-cleanup` | Any OpenAI-compatible `/chat/completions` endpoint | Same cleanup contract, including the same best-effort `Hey Computer` semantics in the shipped prompt, no GPU required. Pre-configured for `https://api.openai.com/v1` + `gpt-4o-mini`; just point it elsewhere if you prefer another provider. |

All three use the **commented-defaults** form: every key is shipped
commented out, with the dataclass default tracked automatically. Lines
you uncomment are overrides for that profile only — future updates that
tweak a default flow through unless you've taken ownership of the line.

To download / install the local model interactively:

```sh
justsayit setup-llm
```

## OpenAI-compatible endpoint

Activate `openai-cleanup` from the tray and you're done — the profile
ships with `endpoint = "https://api.openai.com/v1"` and
`model = "gpt-4o-mini"` already uncommented. Provide the API key via
[any of the three resolver tiers](configuration.md#api-keys-env), then
flip postprocess on. Remote requests retry transient failures by default
with `remote_retries = 3` and `remote_retry_delay_seconds = 1.0`.

Works with anything that speaks the OpenAI chat-completions schema:
OpenAI, OpenRouter, Groq, Together, vLLM, Ollama (`/v1`), LM Studio,
llama.cpp's bundled server, etc. To switch provider:

```toml
# ~/.config/justsayit/postprocess/openai-cleanup.toml
endpoint = "https://api.groq.com/openai/v1"
model = "llama-3.3-70b-versatile"
# api_key_env = "GROQ_API_KEY"   # default OPENAI_API_KEY also fine
```

When `endpoint` is set AND `system_prompt` is left at the dataclass
default (commented out), justsayit auto-swaps the Gemma `<|think|>`
channel cleanup prompt for a channel-free variant. Generic models don't
have that channel and would otherwise reply literally `No changes.` or
leak reasoning into the output.

## Personal-context sidecar

`~/.config/justsayit/context.toml` holds a free-form string appended to
every cleanup prompt under a `# User context` heading. Use it to teach
the model your name, country / languages, and any project-specific
spellings:

```toml
context = """
Name: Jane Doe
Country: Germany
Languages: German (native), English (fluent), Python
Notes: software engineer; often dictates code-related text.
"""
```

Lives in its own file so updates to the shipped profile templates never
clobber it. A profile-level `context = "..."` (in the profile TOML)
overrides the sidecar for that one profile.

Dynamic context runs before this static user-context block, so the
prompt order is: dynamic state first, then the normal system prompt,
then `# User context` when configured.

## Custom profiles

A profile is just a TOML file under `postprocess/` — drop one in and
it appears in the tray's profile picker on next launch. Copy
`gemma4-cleanup.toml` or `openai-cleanup.toml` as a starting point,
rename it (e.g. `translate-en.toml`), and override `system_prompt`.

The dataclass keys you can override:

| Key | Purpose |
|-----|---------|
| `system_prompt` | The instruction sent to the model. Multi-line strings welcome. |
| `temperature` | Lower = deterministic (cleanup); higher = creative (emoji, rewriting). |
| `max_tokens` | Hard cap on the generated reply. |
| `user_template` | Template wrapping the transcript. `{text}` is substituted. |
| `paste_strip_regex` | Regex (`re.DOTALL`) applied to the LLM output before paste but not before overlay display. Useful to hide reasoning preambles. |
| `context` | Per-profile context that overrides the sidecar. |
| **Local LLM** | `model_path`, `hf_repo`, `hf_filename`, `n_gpu_layers`, `n_ctx` |
| **Remote LLM** | `endpoint`, `model`, `api_key`, `api_key_env`, `request_timeout`, `remote_retries`, `remote_retry_delay_seconds` |

### Custom-prompt examples

**Emojify** (the shipped `gemma4-fun.toml`):

```toml
temperature = 0.4
paste_strip_regex = ""
system_prompt = """
Emojify the transcript as much as possible. Keep the original wording
and order, just sprinkle in plenty of fitting emojis. Reply with the
emojified text only — no explanations, no preamble.
"""
```

**Translate to English** (drop into `postprocess/translate-en.toml`):

```toml
temperature = 0.1
paste_strip_regex = ""
system_prompt = """
Translate the user's transcript to natural English. If it's already in
English, return it unchanged. Reply with the translation only — no
preamble, no quotes.
"""
```

**Summarise to bullet points**:

```toml
temperature = 0.2
paste_strip_regex = ""
system_prompt = """
Summarise the user's transcript as a tight bullet list (max 5 items).
Reply with the bullets only — no preamble, no closing remark.
"""
```

**Style change — formal email tone**:

```toml
temperature = 0.3
paste_strip_regex = ""
system_prompt = """
Rewrite the user's transcript in a polite, formal email tone. Preserve
the meaning and key facts; remove filler words. Reply with the rewritten
text only.
"""
```

**Switch to a remote endpoint** for any of the above by adding:

```toml
endpoint = "https://api.openai.com/v1"
model = "gpt-4o-mini"
# api_key_env = "OPENAI_API_KEY"   # default
```

Switch profiles at runtime from the tray's *Postprocess profile* submenu —
the active profile is persisted to `state.toml` so it survives a restart.

## "Thinking" overlay

When a profile's `paste_strip_regex` matches part of the LLM output,
the matched substring is shown above the cleaned text in the overlay
(but stripped from what gets pasted). The default Gemma profile uses
this to surface the model's reasoning channel as a preamble — handy
during iteration, invisible in the pasted result. Disable by setting
`paste_strip_regex = ""`.
