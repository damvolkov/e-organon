"""tests/unit/e_organon/providers/adapters/media/ffmpeg: a real container probed and decoded by the system's own ffmpeg —
the length and whether it carries sound or picture, the audio out as a mono wav at the asked rate and in the asked window,
the pictures out one per interval at the asked size, and the refusals the tools voice."""

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


async def test_ffmpeg_probes_whether_there_is_a_picture(adapter: FfmpegAdapter, tmp_path: Path) -> None:
    await _write(tmp_path / "tone.wav", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "8000")
    await _write(tmp_path / "red.mp4", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1:r=5", "-an")
    sound, picture = await adapter.probe(tmp_path / "tone.wav"), await adapter.probe(tmp_path / "red.mp4")
    assert (sound.has_video, picture.has_video) == (False, True)


@pytest.mark.parametrize(
    ("start", "length", "seconds"),
    [(1.0, 1.0, 1), (0.0, 2.0, 2), (3.0, None, 1)],
    ids=["a-window", "the-first-seconds", "from-a-start-to-the-end"],
)
async def test_ffmpeg_extracts_only_the_window_asked(
    adapter: FfmpegAdapter, tmp_path: Path, start: float, length: float | None, seconds: int
) -> None:
    await _write(tmp_path / "tone.wav", "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-ar", "8000")
    data = b"".join([chunk async for chunk in adapter.extract(tmp_path / "tone.wav", start=start, length=length)])
    back = tmp_path / "back.wav"
    await anyio.Path(back).write_bytes(data)
    assert round((await adapter.probe(back)).duration) == seconds


async def test_ffmpeg_frames_one_picture_every_interval_at_the_asked_size(
    adapter: FfmpegAdapter, tmp_path: Path
) -> None:
    await _write(tmp_path / "red.mp4", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=3:r=5", "-an")
    seen = [frame async for frame in adapter.frames(tmp_path / "red.mp4", every=1.0, width=32, height=16)]
    assert [(frame.at, len(frame.pixels)) for frame in seen] == [
        (0.0, 32 * 16 * 3),
        (1.0, 32 * 16 * 3),
        (2.0, 32 * 16 * 3),
    ]
    ### packed bgr24: red is the last byte of a pixel.
    blue, green, red = seen[0].pixels[:3]
    assert (red > 200, blue < 60, green < 60) == (True, True, True)


@pytest.mark.parametrize("kind", ["extract", "frames"])
async def test_ffmpeg_a_decode_that_fails_is_refused_with_the_childs_reason(
    adapter: FfmpegAdapter, tmp_path: Path, kind: str
) -> None:
    junk = tmp_path / "junk.mp4"
    await anyio.Path(junk).write_bytes(b"not a container at all")
    stream = adapter.extract(junk) if kind == "extract" else adapter.frames(junk, every=1.0, width=8, height=8)
    with pytest.raises(MediaError, match=r"junk\.mp4"):
        [_ async for _ in stream]


async def test_ffmpeg_frames_of_a_file_with_no_picture_are_refused(adapter: FfmpegAdapter, tmp_path: Path) -> None:
    await _write(tmp_path / "tone.wav", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "8000")
    with pytest.raises(MediaError):
        [_ async for _ in adapter.frames(tmp_path / "tone.wav", every=1.0, width=8, height=8)]


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


async def test_ffmpeg_a_child_gone_quiet_past_the_stall_is_killed_and_refused(tmp_path: Path) -> None:
    mute = tmp_path / "mute.sh"
    mute.write_text("#!/bin/sh\nexec sleep 30\n")
    mute.chmod(0o755)
    quiet = FfmpegAdapter(timeout=timedelta(seconds=30), stall=timedelta(seconds=0.3), ffmpeg=str(mute))
    ### killed, not waited for: the refusal arrives long before the child would have exited by itself.
    with anyio.fail_after(5), pytest.raises(MediaError, match="went quiet"):
        _ = [chunk async for chunk in quiet.extract(tmp_path / "any.wav")]
