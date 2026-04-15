# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
