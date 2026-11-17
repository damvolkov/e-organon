"""tests/unit/e_organon/providers/adapters/media/ffmpeg: a real container probed and decoded by the system's own ffmpeg —
the length and whether it carries sound, the audio out as a mono wav at the asked rate, and the refusals the tools voice."""

from datetime import timedelta
from typing import TYPE_CHECKING

import anyio
import pytest

from e_organon.infra.errors import MediaError
from e_organon.providers.adapters.media.ffmpeg import FfmpegAdapter

if TYPE_CHECKING:
    from pathlib import Path


async def _write(path: Path, *order: str) -> None:
    """A container written by the same binary under test, so no fixture denies the probe."""
    done = await anyio.run_process(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *order, str(path), "-y"], check=False
    )
    assert done.returncode == 0, done.stderr


async def test_ffmpeg_probes_the_length_and_the_sound(adapter: FfmpegAdapter, tmp_path: Path) -> None:
    await _write(tmp_path / "tone.wav", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "8000")
    info = await adapter.probe(tmp_path / "tone.wav")
    assert (round(info.duration), info.has_audio) == (2, True)


async def test_ffmpeg_a_file_with_no_audio_has_none_to_give(adapter: FfmpegAdapter, tmp_path: Path) -> None:
    await _write(tmp_path / "black.mp4", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-an")
    assert (await adapter.probe(tmp_path / "black.mp4")).has_audio is False


async def test_ffmpeg_extracts_a_streamed_mono_wav(adapter: FfmpegAdapter, tmp_path: Path) -> None:
    await _write(tmp_path / "tone.wav", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "8000")
    data = b"".join([chunk async for chunk in adapter.extract(tmp_path / "tone.wav", sample_rate=16_000)])
    assert (data[:4], data[8:12]) == (b"RIFF", b"WAVE")
    back = tmp_path / "back.wav"
    await anyio.Path(back).write_bytes(data)
    info = await adapter.probe(back)
    assert (info.has_audio, round(info.duration)) == (True, 1)


async def test_ffmpeg_refuses_a_file_that_is_not_media(adapter: FfmpegAdapter, tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    await anyio.Path(junk).write_bytes(b"not a container at all")
    with pytest.raises(MediaError):
        await adapter.probe(junk)


async def test_ffmpeg_is_healthy_where_the_binaries_answer(adapter: FfmpegAdapter) -> None:
    assert await adapter.health() is True


@pytest.fixture
def adapter() -> FfmpegAdapter:
    return FfmpegAdapter(timeout=timedelta(seconds=30), stall=timedelta(seconds=30))
