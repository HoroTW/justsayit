# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`justsayit` is a local Parakeet-powered voice dictation app for Wayland.
It is a single-process GTK4 app with a layer-shell overlay. ASR backends
(Parakeet / Whisper / OpenAI-compatible) feed an optional LLM cleanup pass
(local llama-cpp-python or OpenAI-compatible endpoints).

## Common Commands

```sh
# Run the app after install.sh has set up the venv
uv run justsayit
uv run justsayit --no-overlay
uv run justsayit --no-paste
uv run justsayit init
uv run justsayit show-defaults config

# Tests
uv run pytest
uv run pytest --no-network
uv run pytest -m burn
uv run pytest tests/test_postprocess.py::test_name -v

# Version bump
./scripts/bump-version.py 0.13.21
```

Every commit should bump the version and add a `CHANGELOG.md` entry, even
for small fixes. Use `scripts/bump-version.py`; do not manually edit only
one version file.

## Sandbox Notes

In restricted sandboxes, importing `sounddevice` can hang. Because
`justsayit.audio` imports `sounddevice`, any test module that imports the
audio engine can also hang during collection.

Observed symptoms:

- `python -c "import sounddevice; print('ok')"` times out with no output.
- `python -c "import justsayit.audio; print('ok')"` times out with no output.
- `pytest tests/test_audio.py -q` hangs during collection or emits no output.
- The same commands complete normally outside the restricted sandbox.

If this happens, do not treat it as a test failure in the code under test.
Rerun the focused test outside the sandbox / with the approved escalation
path and report that the sandboxed import path is the reason for the hang.

## Architecture Pointers

- `src/justsayit/audio.py`: `AudioEngine`, capture state machine, VAD,
  streaming partial chunks, debug WAV dumping.
- `src/justsayit/pipeline.py`: `SegmentPipeline.handle()`, the
  transcribe -> filter -> LLM -> paste flow.
- `src/justsayit/cli.py`: `App`, queue wiring, transcribe worker, UI
  callbacks.
- `src/justsayit/transcribe_*.py`: one file per transcription backend.
- `src/justsayit/postprocess/`: LLM cleanup backends and profile loading.
- `src/justsayit/config/`: dataclass schema and config I/O.

Cross-thread UI updates must go through `GLib.idle_add`, generally via
`OverlayWindow.push_*` helpers.

## Audio State

`AudioEngine` has four states: `IDLE`, `VALIDATING`, `RECORDING`,
`MANUAL`.

- `vad.enabled = false`: hotkey-only. The mic is closed until
  `start_manual()` opens it; `stop_manual()` emits the segment.
- `vad.enabled = true`: Silero VAD watches a continuously-open stream.
  The first `validation_seconds` are transcribed as a "did we hear words?"
  check before transitioning to `RECORDING`.

Long recordings may emit `stream-chunk` partial segments during capture.
Filter, LLM, and paste still run once on the final combined text.

## Coding Style

- Keep changes surgical and tied to the user request.
- Prefer existing project patterns over new abstractions.
- Do not refactor unrelated code.
- Add focused regression tests for bugs.
- When fixing prompts under `src/justsayit/prompts/*.md`, add general
  rules rather than copying a specific failing user sentence.
- Backend and transcriber modules are intentionally self-contained.
  Similar code across independently replaceable modules is acceptable;
  only extract shared helpers for truly identical infrastructure.
