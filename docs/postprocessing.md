# LLM postprocessing

An optional LLM cleanup pass runs after transcription. The default
"cleanup" profile fixes obvious mishears, removes filler words, applies
dictated formatting / punctuation, and only switches into assistant mode
when the transcript starts with `Hey Computer`. The same machinery can
do anything else you'd ask an LLM to do — emojify, translate,
summarise, change tone, format as Markdown, etc. — by swapping in a
custom system prompt.

Enable in `config.toml` (or toggle from the tray):

```toml
[postprocess]
enabled = true
profile = "gemma4-cleanup"   # filename stem under postprocess/
```

## Shipped profiles

`justsayit init` writes three profiles into `~/.config/justsayit/postprocess/`:

| Profile | Backend | What it does |
|---------|---------|--------------|
| `gemma4-cleanup` | Local Gemma 4 E4B via `llama-cpp-python` | Recommended. Conservative DE/EN cleanup tuned for Gemma. Switches into assistant mode on a leading `Hey Computer`. |
| `gemma4-fun` | Same local Gemma model | Keeps your wording but sprinkles emojis. Great for chat / social. |
| `openai-cleanup` | Any OpenAI-compatible `/chat/completions` endpoint | Same cleanup contract, no GPU required. Pre-configured for `https://api.openai.com/v1` + `gpt-4o-mini`; just point it elsewhere if you prefer another provider. |

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
flip postprocess on.

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
| **Remote LLM** | `endpoint`, `model`, `api_key`, `api_key_env`, `request_timeout` |

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
