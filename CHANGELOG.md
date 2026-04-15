# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
