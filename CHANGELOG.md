# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-04-15

### Added

- **LLM postprocessor** — optional cleanup step that runs after ASR and
  regex filters. Uses a local GGUF model via `llama-cpp-python` to
  remove filler words, fix grammar/spelling, and correct misheard words
  while preserving meaning and style.
- **`[postprocess]` config section** — `enabled` (default `false`) and
  `profile` (name of the profile file to load, default `"gemma-cleanup"`).
- **Per-model profile files** — each LLM is configured in its own TOML
  file at `~/.config/justsayit/postprocess/<name>.toml`. Settings:
  `model_path`, `hf_repo` + `hf_filename` (for auto-download),
  `n_gpu_layers` (`-1` = all on GPU), `n_ctx`, `temperature`,
  `max_tokens`, `system_prompt`, `user_template`.
- **Default profile `gemma-cleanup.toml`** written by `justsayit init`
  with a German system prompt and `temperature = 0.08` (deterministic).
- **Auto-download** — if `hf_repo` + `hf_filename` are set in the
  profile and `model_path` doesn't exist, `justsayit download-models`
  fetches the GGUF from HuggingFace.
- **`[llm]` install extra** — `pip install 'justsayit[llm]'` for CPU.
  For Vulkan GPU (AMD/Intel): `CMAKE_ARGS="-DGGML_VULKAN=1" pip install
  llama-cpp-python`.
- **`install.sh --postprocess`** — compiles and installs
  `llama-cpp-python` with `GGML_VULKAN=1`; validates that cmake and
  Vulkan headers are present before starting; then launches the
  interactive `setup-llm` wizard.
- **`justsayit setup-llm`** — interactive wizard that lists the built-in
  model catalogue (gemma4, qwen3-4b, qwen3-0.8b), queries the
  HuggingFace API for the Q4_K_M GGUF filename, downloads it to the
  local cache, and patches the profile to point at the downloaded file.
  Pass `--model KEY` to skip the interactive prompt.

## [0.4.0] - 2026-04-15

### Added

- **Multi-backend transcription** — `model.backend` can now be set to
  `"parakeet"` (default, sherpa-onnx, bundled dep) or `"whisper"`
  (faster-whisper / distil-whisper, optional dep).
- **`model.whisper_model`** — HuggingFace model ID or local path for the
  Whisper backend (default: `"Systran/faster-distil-whisper-large-v3"`).
- **`model.whisper_device`** — inference device for Whisper (`"cpu"` or
  `"cuda"`, default `"cpu"`).
- **`model.whisper_compute_type`** — CTranslate2 quantisation for Whisper
  (`"int8"`, `"float16"`, `"float32"`, default `"int8"`).
- **`[whisper]` install extra** — `uv pip install 'justsayit[whisper]'`
  (or `install.sh --model whisper`) pulls in `faster-whisper`.
- **`install.sh --model parakeet|whisper`** — select backend at install
  time; writes `model.backend` into config.toml and installs required extras.
- **`justsayit init --backend parakeet|whisper`** — set backend in the
  generated config.toml without editing it by hand.
- Whisper model downloads lazily from HuggingFace on first transcription
  into `<cache>/justsayit/models/whisper/`; no extra download step needed.

### Changed

- `install.sh`: `gtk4-layer-shell` is now a **hard install blocker** (was
  a warning). The Wayland layer-shell overlay cannot work without it, so
  aborting early gives a clearer error message.
- `justsayit download-models` now prints a tailored message for the Whisper
  backend (only downloads the tiny VAD ONNX; Whisper model is deferred).

## [0.3.3] - 2026-04-15

### Fixed

- Auto-listen tray toggle now works immediately after the first reload.
  Previously, starting with `vad.enabled = false` meant the VAD model was
  never loaded (`vad_loaded = false`), so `vad_enabled` stayed `false`
  regardless of tray clicks — the checkbox appeared stuck and only a second
  reload (which picked up the silently-saved `vad.enabled = true`) fixed it.
  The VAD model is now always loaded on startup since it is always downloaded.

## [0.3.2] - 2026-04-15

### Added

- **Mute / unmute sounds** for VAD auto-listen mode — a descending two-tone
  "dub-di" (G4 → D4) plays when VAD is paused via the hotkey, and an
  ascending "dub-do" (D4 → G4) plays when it is resumed.

## [0.3.1] - 2026-04-15

### Changed

- In VAD auto-listen mode the start chime now plays as soon as the overlay
  appears (entering `VALIDATING`) at a reduced volume, giving early auditory
  feedback while the result is still uncertain. The scale is configurable via
  `sound.validating_volume_scale` (default `0.4`).
- The stop chime now plays whenever the overlay disappears (any → `IDLE`),
  including validation failures and manual stops.
- Hotkey-triggered (manual) recordings still play the start chime at full
  volume.

## [0.3.0] - 2026-04-15

### Added

- **Notification sounds** — a short chime plays when recording starts (A4,
  380 ms) and a lower, longer chime when it stops (E4, 530 ms). Sounds are
  generated from first-principles using numpy and bundled as WAV files; no
  external assets required.
- **`sound.enabled`** — master switch to disable sounds entirely.
- **`sound.volume`** — playback volume (0.0–1.0, default 1.0).
- `sounds/generate_sounds.py` — developer script to regenerate the bundled
  WAV files (not needed by end-users).

## [0.2.2] - 2026-04-15

### Added

- **Reload config** tray menu item — restarts the process via `execve` so
  all config changes (including overlay, audio, and model settings that
  cannot be hot-reloaded) take effect immediately.

## [0.2.1] - 2026-04-15

### Changed

- Default overlay width reduced from 260 to 174 (⅔ of previous).

## [0.2.0] - 2026-04-15

### Added

- **`paste.auto_space_timeout_ms`** — prepend a space before a transcription
  when the previous one finished within this many milliseconds, so continuous
  dictation works without manually inserting spaces between phrases. The
  timeout is checked against when the new recording *started* (derived from
  segment duration), so long recordings never incorrectly skip the prefix.
- **`paste.append_trailing_space`** — always append a trailing space after
  every transcription so the cursor is ready for the next word. Takes
  precedence over `auto_space_timeout_ms` when both are set; a desktop
  notification warns about the conflict.
- **`overlay.visualizer_sensitivity`** — scale factor for the mic-level bar
  (default `1.0`). Increase if your microphone records quietly; decrease if
  the bar clips on every word.
- **`overlay.opacity`** — background opacity of the overlay pill (`0.0`–`1.0`,
  default `0.78`). Applied uniformly to the entire widget (background, text,
  dot, and meter) via GTK `set_opacity`.

### Changed

- Overlay layout: status dot is now on the left and vertically centered;
  state label sits above the level meter in a vertical stack on the right.
- Visualizer bar grows symmetrically from the center outward instead of
  filling left-to-right.

## [0.1.0] - 2026-04-13

### Added

- Initial release.
