"""Model auto-download and path resolution.

Fetches the sherpa-onnx Parakeet TDT v3 INT8 bundle and the Silero VAD
ONNX file into the user cache directory. No huggingface-hub dependency:
plain ``urllib`` with a progress indicator is enough for two files.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from justsayit.config import Config, models_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPaths:
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path
    vad: Path


def _download(url: str, dest: Path, *, chunk: int = 1 << 15) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "justsayit/0.1"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -1
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                pct = int(done * 100 / total)
                if pct != last_pct and sys.stderr.isatty():
                    sys.stderr.write(f"\r  {dest.name}: {pct}% ({done // 1024} KiB)")
                    sys.stderr.flush()
                    last_pct = pct
        if total and sys.stderr.isatty():
            sys.stderr.write("\n")
    tmp.replace(dest)


def _extract_tar(archive: Path, out_dir: Path) -> None:
    log.info("extracting %s -> %s", archive, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        # Basic safety: refuse entries that would escape out_dir.
        safe_members = []
        for m in tf.getmembers():
            resolved = (out_dir / m.name).resolve()
            if not str(resolved).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
            safe_members.append(m)
        tf.extractall(out_dir, members=safe_members)


def _parakeet_dir(cfg: Config) -> Path:
    return models_dir() / cfg.model.parakeet_archive_dir


def _vad_path(cfg: Config) -> Path:
    return models_dir() / "silero_vad.onnx"


def paths(cfg: Config) -> ModelPaths:
    """Resolve ModelPaths without checking existence."""
    base = _parakeet_dir(cfg)
    return ModelPaths(
        encoder=base / cfg.model.parakeet_encoder,
        decoder=base / cfg.model.parakeet_decoder,
        joiner=base / cfg.model.parakeet_joiner,
        tokens=base / cfg.model.parakeet_tokens,
        vad=_vad_path(cfg),
    )


def ensure_models(cfg: Config, *, force: bool = False, want_vad: bool = True) -> ModelPaths:
    """Download any missing model files. Returns resolved ModelPaths.

    ``want_vad=False`` skips the tiny Silero download when VAD is
    disabled; the ``vad`` path in the result still points at the
    expected location but the file may not exist.
    """
    p = paths(cfg)
    base = _parakeet_dir(cfg)

    parakeet_complete = all(
        x.exists() for x in (p.encoder, p.decoder, p.joiner, p.tokens)
    )
    if force or not parakeet_complete:
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "parakeet.tar.bz2"
            _download(cfg.model.parakeet_archive_url, archive)
            # Stage into temp dir, then move the expected sub-dir into models_dir.
            staged = Path(td) / "unpacked"
            _extract_tar(archive, staged)
            src_dir = staged / cfg.model.parakeet_archive_dir
            if not src_dir.is_dir():
                # Some archives may unpack into a different top-level name;
                # fall back to the single top-level entry if present.
                entries = [e for e in staged.iterdir() if e.is_dir()]
                if len(entries) == 1:
                    src_dir = entries[0]
                else:
                    raise RuntimeError(
                        f"Could not find model directory in archive; "
                        f"expected {cfg.model.parakeet_archive_dir!r} "
                        f"inside {staged}"
                    )
            if base.exists():
                shutil.rmtree(base)
            shutil.move(str(src_dir), str(base))

    if want_vad and (force or not p.vad.exists()):
        _download(cfg.model.vad_url, p.vad)

    required = [p.encoder, p.decoder, p.joiner, p.tokens]
    if want_vad:
        required.append(p.vad)
    missing = [x for x in required if not x.exists()]
    if missing:
        raise RuntimeError(
            "Model files still missing after download: "
            + ", ".join(str(m) for m in missing)
        )
    return p
