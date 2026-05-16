"""Burn tests for generated speech fixtures.

These fixtures are synthetic (eSpeak NG + ffmpeg), so they are safe to keep in
git. They target known ASR/chunking failure modes where unit tests with stub
transcribers are not enough:

* mid-sentence preview splits can invent punctuation/wording artifacts;
* noisy pauses near the chunk boundary must keep the tail;
* long continuous-ish speech must keep late content after rechunking.
* clipped and auto-gain/noise-floor input should preserve key phrases.
"""

from __future__ import annotations

import wave
import time
from pathlib import Path

import numpy as np
import pytest

from justsayit.config import Config
from justsayit.audio import Segment
from justsayit.pipeline import SegmentPipeline
from justsayit.transcribe_parakeet import ParakeetTranscriber

pytestmark = pytest.mark.burn

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"


class _PreviewOverlay:
    def __init__(self) -> None:
        self.word_previews: list[tuple[str, str, str]] = []
        self.chunk_previews: list[tuple[str, str]] = []
        self.finalized_chunks: list[str] = []
        self.detected: list[tuple[str, bool]] = []

    def push_word_preview_text(
        self,
        finalized_text: str,
        chunk_preview_text: str,
        word_preview_text: str,
    ) -> None:
        self.word_previews.append((finalized_text, chunk_preview_text, word_preview_text))

    def push_chunk_preview_text(self, committed_text: str, preview_text: str) -> None:
        self.chunk_previews.append((committed_text, preview_text))

    def push_finalized_chunk_text(self, text: str) -> None:
        self.finalized_chunks.append(text)

    def push_detected_text(self, text: str, llm_pending: bool = False) -> None:
        self.detected.append((text, llm_pending))

    def push_hide(self) -> None:
        pass

    def push_linger_start(self) -> None:
        pass


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    dtype = np.int16 if sampwidth == 2 else np.int32
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        samples = samples[::n_channels]
    samples /= float(np.iinfo(dtype).max)
    return samples, sr


@pytest.fixture(scope="module")
def transcriber() -> ParakeetTranscriber:
    cfg = Config()
    cfg.model.backend = "parakeet"
    cfg.model.parakeet_normalize = "A"
    cfg.model.parakeet_trim_silence_rms = 0.005
    t = ParakeetTranscriber(cfg)
    if not t.paths.encoder.exists():
        pytest.skip(f"Parakeet model not downloaded (missing {t.paths.encoder})")
    t.warmup()
    return t


@pytest.mark.parametrize("basename,expected", [
    ("synthetic_noisy_pause_chunking", ["silver lantern", "blue notebook"]),
    ("synthetic_long_continuous_dictation", ["preview fragments", "final answer", "hard boundary"]),
    ("synthetic_clipped_mic", ["crimson marker", "green folder"]),
    ("synthetic_variable_gain_noise", ["violet cabinet", "silver window"]),
    ("synthetic_noisy_boundary_shift", [
        "boundary phrase says silver lantern",
        "blue notebook",
        "not delete the blue notebook",
        "copper meadow",
    ]),
])
def test_generated_fixture_keeps_key_content(
    basename: str,
    expected: list[str],
    transcriber: ParakeetTranscriber,
) -> None:
    wav_path = FIXTURES_DIR / f"{basename}.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    out = transcriber.transcribe(samples, sr).strip().lower()

    missing = [phrase for phrase in expected if phrase not in out]
    assert not missing, f"missing {missing!r} from transcript: {out!r}"


def test_mid_sentence_preview_split_has_real_asr_artifacts(
    transcriber: ParakeetTranscriber,
) -> None:
    wav_path = FIXTURES_DIR / "synthetic_mid_sentence_preview.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    duration = len(samples) / float(sr)

    full = transcriber.transcribe(samples, sr).strip().lower()
    first = transcriber.transcribe(samples[:int(6.0 * sr)], sr).strip()
    second = transcriber.transcribe(samples[int(6.0 * sr):], sr).strip()
    preview_joined = f"{first} {second}".strip().lower()

    assert "final instruction stays" in full
    assert "final text" in full
    assert "final instruction stays" not in preview_joined
    assert "final instructions" in preview_joined
    assert "final text" not in preview_joined
    assert duration > 10.0


def test_known_gap_compressed_continuous_keeps_all_boundary_phrases(
    transcriber: ParakeetTranscriber,
) -> None:
    wav_path = FIXTURES_DIR / "synthetic_compressed_continuous_gap.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    out = transcriber.transcribe(samples, sr).strip().lower()

    for phrase in (
        "violet anchor phrase",
        "copper gateway phrase",
        "silver horizon phrase",
        "orange lantern phrase",
    ):
        assert phrase in out, f"missing {phrase!r} from transcript: {out!r}"


def test_long_pause_preview_fixture_does_not_hallucinate_tail(
    transcriber: ParakeetTranscriber,
) -> None:
    wav_path = FIXTURES_DIR / "long_pause_preview_hallucination.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    full = transcriber.transcribe(samples, sr).strip().lower()
    early_preview = transcriber.transcribe_preview(samples[:int(3.0 * sr)], sr).strip().lower()

    assert "i think" in early_preview
    assert "that's the same" not in early_preview
    assert "i think we should do something" in full
    assert "or shouldn't we" in full


def test_streaming_chunk_boundary_does_not_invent_sentence_break(
    transcriber: ParakeetTranscriber,
    capsys,
) -> None:
    wav_path = FIXTURES_DIR / "chunk_boundary_sentence_split_de.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    preview_seconds = 1.2
    step = int(preview_seconds * sr)
    preview_chunks = [
        samples[start:min(len(samples), start + step)]
        for start in range(0, len(samples), step)
    ]

    cfg = Config()
    pipe = SegmentPipeline(cfg, transcriber, filters=[], paster=None, no_paste=True)
    pipe.overlay = _PreviewOverlay()
    for chunk in preview_chunks[:-1]:
        pipe.handle(
            Segment(chunk, sr, reason="stream-chunk", is_final=False),
            is_continue=False,
        )
    pipe.handle(Segment(preview_chunks[-1], sr, reason="manual"), is_continue=False)

    final = capsys.readouterr().out.strip().lower()
    assert "weil ja, ganz grundsätzlich ist das natürlich immer so eine geschichte" in final
    assert "natürlich. immer so eine geschichte" not in final


def test_preview_plus_final_transcription_speed(
    transcriber: ParakeetTranscriber,
    record_property,
    capsys,
) -> None:
    wav_path = FIXTURES_DIR / "synthetic_long_continuous_dictation.wav"
    if not wav_path.exists():
        pytest.skip(f"fixture WAV not present: {wav_path}")

    samples, sr = _load_wav(wav_path)
    preview_seconds = 1.2
    min_preview_seconds = 1.0
    step = int(preview_seconds * sr)
    preview_chunks = [
        samples[start:min(len(samples), start + step)]
        for start in range(0, len(samples), step)
        if (min(len(samples), start + step) - start) / float(sr) >= min_preview_seconds
    ]

    cfg = Config()
    pipe = SegmentPipeline(cfg, transcriber, filters=[], paster=None, no_paste=True)
    overlay = _PreviewOverlay()
    pipe.overlay = overlay
    t0 = time.perf_counter()
    for chunk in preview_chunks[:-1]:
        pipe.handle(
            Segment(chunk, sr, reason="stream-chunk", is_final=False),
            is_continue=False,
        )
    preview_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipe.handle(Segment(preview_chunks[-1], sr, reason="manual"), is_continue=False)
    final_elapsed = time.perf_counter() - t0
    capsys.readouterr()

    recording_seconds = len(samples) / float(sr)
    total_elapsed = preview_elapsed + final_elapsed
    final_stop_ratio = final_elapsed / recording_seconds
    preview_plus_final_wall_ratio = total_elapsed / recording_seconds

    record_property("recording_seconds", recording_seconds)
    record_property("preview_chunks", len(preview_chunks))
    record_property("word_preview_updates", len(overlay.word_previews))
    record_property("chunk_preview_updates", len(overlay.chunk_previews))
    record_property("finalized_chunk_updates", len(overlay.finalized_chunks))
    record_property("preview_elapsed_seconds", preview_elapsed)
    record_property("final_stop_elapsed_seconds", final_elapsed)
    record_property("preview_plus_final_elapsed_seconds", total_elapsed)
    record_property("final_stop_wall_ratio", final_stop_ratio)
    record_property("preview_plus_final_wall_ratio", preview_plus_final_wall_ratio)

    assert final_stop_ratio < 0.20, (
        f"final stop ASR too slow: {final_elapsed:.2f}s for "
        f"{recording_seconds:.2f}s recording (ratio={final_stop_ratio:.3f})"
    )
    assert preview_plus_final_wall_ratio < 0.75, (
        f"preview+final ASR too slow: {total_elapsed:.2f}s for "
        f"{recording_seconds:.2f}s recording "
        f"(ratio={preview_plus_final_wall_ratio:.3f}, chunks={len(preview_chunks)})"
    )
