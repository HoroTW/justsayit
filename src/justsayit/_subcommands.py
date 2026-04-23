"""Subcommand implementations for the justsayit CLI.

These functions are called from ``cli.main()`` but have no dependency on
the ``App`` class or GTK — they can be imported without loading the UI.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from justsayit.config import (
    Config,
    cache_dir,
    config_dir,
    ensure_config_file,
    ensure_dirs,
    ensure_filters_file,
    load_config,
)
from justsayit.postprocess import (
    KNOWN_LLM_MODELS,
    LLMPostprocessor,
    _CLEANUP_PROFILE_TOML,
    apply_profile_overrides,
    download_llm_model,
    ensure_default_profiles,
    ensure_dynamic_context_script,
    ensure_profile,
    find_hf_q4_filename,
    load_profile,
    profiles_dir,
    update_profile_model,
)

log = logging.getLogger("justsayit")


def _write_default_config(force: bool = False, backend: str | None = None) -> None:
    ensure_dirs()
    cfg_path = config_dir() / "config.toml"
    filters_path = config_dir() / "filters.json"

    cfg_pre_existed = cfg_path.exists()
    if cfg_pre_existed and force:
        cfg_path.unlink()
        cfg_pre_existed = False
    ensure_config_file(cfg_path)
    if cfg_pre_existed:
        print(f"config already exists: {cfg_path}", file=sys.stderr)
    else:
        if backend is not None and backend != Config().model.backend:
            with cfg_path.open("a", encoding="utf-8") as f:
                f.write(f'\n[model]\nbackend = "{backend}"\n')
        print(f"wrote {cfg_path}")

    cleanup_path, fun_path, openai_path, ollama_gemma_path = ensure_default_profiles()
    dynamic_context_path = ensure_dynamic_context_script()
    print(f"postprocess profile: {cleanup_path}    (recommended)")
    print(f"postprocess profile: {fun_path}        (emoji-heavy variant)")
    print(f"postprocess profile: {openai_path}     (OpenAI-compatible endpoint)")
    print(f"postprocess profile: {ollama_gemma_path}  (Ollama-served Gemma)")
    print(f"dynamic-context script: {dynamic_context_path}")

    filters_pre_existed = filters_path.exists()
    if filters_pre_existed and force:
        filters_path.unlink()
        filters_pre_existed = False
    ensure_filters_file(filters_path)
    if filters_pre_existed:
        print(f"filters already exist: {filters_path}", file=sys.stderr)
    else:
        print(f"wrote {filters_path}")


def _download_models_only() -> int:
    ensure_dirs()
    cfg = load_config()
    if cfg.model.backend == "openai":
        from justsayit.model import ensure_vad
        vad = ensure_vad(cfg, force=False)
        print(f"openai backend — VAD model ready: {vad}")
        print(
            f"  (transcription served by {cfg.model.openai_endpoint or '<unset endpoint>'})"
        )
    elif cfg.model.backend == "whisper":
        from justsayit.model import ensure_vad
        vad = ensure_vad(cfg, force=False)
        print(f"whisper backend — VAD model ready: {vad}")
        print("  (Whisper model downloads automatically on first transcription)")
    else:
        from justsayit.model import ensure_models
        p = ensure_models(cfg, force=False)
        print(f"models ready:\n  encoder: {p.encoder}\n  vad:     {p.vad}")

    if cfg.postprocess.enabled:
        try:
            profile = load_profile(cfg.postprocess.profile)
            if profile.endpoint:
                print(f"LLM endpoint: {profile.endpoint} (model={profile.model!r})")
            elif profile.hf_repo and profile.hf_filename:
                pp = LLMPostprocessor(profile)
                model_path = pp._resolved_model_path()
                print(f"LLM model ready: {model_path}")
        except Exception as exc:
            print(f"postprocess download skipped: {exc}", file=sys.stderr)
    return 0


def _setup_file_logging(cfg: Config, console_level: int) -> None:
    """Attach a rotating file handler if ``log.file_enabled=True``."""
    root = logging.getLogger()
    for h in root.handlers:
        if not isinstance(h, logging.handlers.RotatingFileHandler):
            h.setLevel(console_level)

    if not cfg.log.file_enabled:
        return

    path = cfg.log.file_path.strip() or str(cache_dir() / "justsayit.log")
    resolved = Path(path).expanduser()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            resolved,
            maxBytes=cfg.log.file_max_bytes,
            backupCount=cfg.log.file_backup_count,
            encoding="utf-8",
        )
    except OSError as e:
        log.error("could not open debug log file %s: %s", resolved, e)
        return

    try:
        file_level = getattr(logging, cfg.log.file_level.upper())
    except AttributeError:
        log.warning("unknown log.file_level=%r — falling back to DEBUG", cfg.log.file_level)
        file_level = logging.DEBUG
    handler.setLevel(file_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(min(console_level, file_level))
    log.info(
        "debug log file enabled: %s (level=%s, max_bytes=%d, backups=%d)",
        resolved,
        logging.getLevelName(file_level),
        cfg.log.file_max_bytes,
        cfg.log.file_backup_count,
    )


def _ensure_llama_cpp(vulkan: bool = True) -> bool:
    """Ensure llama-cpp-python is importable, compiling it if needed.

    Returns True when llama_cpp is ready, False on unrecoverable error.
    """
    import shutil
    import subprocess

    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        pass

    if (
        subprocess.run(
            [sys.executable, "-c", "import llama_cpp"],
            capture_output=True,
        ).returncode
        == 0
    ):
        return True

    print("\nllama-cpp-python is not installed — installing now.")

    if vulkan:
        if shutil.which("cmake") is None:
            print(
                "error: cmake is required to compile llama-cpp-python.\n"
                "  Install: pacman -S cmake  (or the equivalent for your distro)",
                file=sys.stderr,
            )
            return False
        vulkan_ok = (
            subprocess.run(
                ["pkg-config", "--exists", "vulkan"], capture_output=True
            ).returncode
            == 0
        )
        if not vulkan_ok:
            print(
                "warning: Vulkan headers not found — falling back to CPU-only build.\n"
                "  For GPU acceleration later: pacman -S vulkan-headers vulkan-icd-loader\n",
                file=sys.stderr,
            )
            vulkan = False

    if vulkan:
        print(
            "Compiling with Vulkan GPU support"
            " (CMAKE_ARGS=-DGGML_VULKAN=1) — this may take several minutes…\n"
        )
        env = {**os.environ, "CMAKE_ARGS": "-DGGML_VULKAN=1"}
    else:
        print("Installing llama-cpp-python (CPU-only build)…\n")
        env = {k: v for k, v in os.environ.items() if k != "CMAKE_ARGS"}

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3"],
        env=env,
    )
    if result.returncode != 0:
        print("error: llama-cpp-python installation failed.", file=sys.stderr)
        return False
    print("\nllama-cpp-python installed successfully.")
    return True


def _parse_selection(raw: str, max_n: int) -> list[int] | None:
    """Parse a selection string into sorted 1-based indices.

    Accepts a single number ("2"), comma-separated ("1,3"), ranges ("1-3"),
    mixed ("1,3-5"), or the keyword "all". Returns None if invalid.
    """
    s = raw.strip().lower()
    if s in ("all", "a"):
        return list(range(1, max_n + 1))
    indices: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            sides = part.split("-", 1)
            try:
                lo, hi = int(sides[0].strip()), int(sides[1].strip())
            except ValueError:
                return None
            if not (1 <= lo <= hi <= max_n):
                return None
            indices.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError:
                return None
            if not (1 <= n <= max_n):
                return None
            indices.add(n)
    return sorted(indices) if indices else None


def _run_setup_llm(model_key: str | None = None, cpu: bool = False) -> int:
    """Interactively select, download, and configure one or more GGUF LLM models."""
    ensure_dirs()

    if not _ensure_llama_cpp(vulkan=not cpu):
        return 1

    keys = list(KNOWN_LLM_MODELS.keys())

    if model_key is not None:
        selected_keys = [model_key]
    else:
        print("\nAvailable LLM models for transcription cleanup:\n")
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {KNOWN_LLM_MODELS[key]['display']}")
        print()
        hint = f"1-{len(keys)}" if len(keys) > 2 else "1,2"
        while True:
            try:
                raw = input(
                    f"Select model(s) [number, range, or comma-separated, e.g. 1 or {hint}]"
                    " (Ctrl-C to skip): "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSkipped.")
                return 0
            indices = _parse_selection(raw, len(keys))
            if indices is not None:
                selected_keys = [keys[i - 1] for i in indices]
                break
            print(
                f"  Enter a number (1-{len(keys)}), a range (1-{len(keys)}),"
                " or comma-separated values."
            )

    downloaded_models: list[tuple[str, Path]] = []
    activate_options: list[str] = []
    failed: list[str] = []

    for key in selected_keys:
        info = KNOWN_LLM_MODELS[key]
        hf_repo = info["hf_repo"]
        print(f"\n{info['display']}")
        print("  Querying HuggingFace for Q4_K_M filename…", end="", flush=True)
        try:
            hf_filename = find_hf_q4_filename(hf_repo)
        except RuntimeError as exc:
            print(f"\n  error: {exc}", file=sys.stderr)
            failed.append(key)
            continue
        print(f" {hf_filename}")

        try:
            model_path = download_llm_model(hf_repo, hf_filename)
        except Exception as exc:
            print(f"  error: download failed: {exc}", file=sys.stderr)
            failed.append(key)
            continue

        downloaded_models.append((key, model_path))
        print(f"  Model:   {model_path}")

        if key == "gemma4":
            cleanup_path, fun_path, _openai_path, _ollama_path = (
                ensure_default_profiles()
            )
            update_profile_model(cleanup_path, model_path, hf_repo, hf_filename)
            update_profile_model(fun_path, model_path, hf_repo, hf_filename)
            activate_options.extend(["gemma4-cleanup", "gemma4-fun"])
            print(f"  Profile: {cleanup_path}")
            print(f"  Profile: {fun_path}")
        else:
            profile_path = profiles_dir() / f"{key}.toml"
            ensure_profile(_CLEANUP_PROFILE_TOML, profile_path)
            update_profile_model(profile_path, model_path, hf_repo, hf_filename)
            overrides = info.get("profile_overrides") or {}
            if overrides:
                apply_profile_overrides(profile_path, overrides)
                print(
                    "  Applied model-specific overrides: "
                    + ", ".join(f"{k}={v}" for k, v in overrides.items())
                )
            activate_options.append(key)
            print(f"  Profile: {profile_path}")

    if not downloaded_models:
        return 1 if failed else 0

    print(f"\n{'─' * 54}")
    if len(downloaded_models) == 1:
        print("\n1 model ready. Available profile(s):")
    else:
        print(f"\n{len(downloaded_models)} model(s) ready. Available profile(s):")
    for name in activate_options:
        print(f"  - {name}")
    print(
        "\nPick one from the tray menu (LLM submenu) — that's a runtime"
        " toggle and writes to ~/.config/justsayit/state.toml for you."
    )
    return 0


def _send_toggle(*, profile: str | None, use_clipboard: bool) -> int:
    from justsayit.toggle_client import send_toggle as _send
    return _send(profile=profile, use_clipboard=use_clipboard)
