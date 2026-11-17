"""tests/unit/e_organon/services/audio/transcribe: audio/transcribe over faked media and engine — the steps in order, the
wav really streamed to the engine, the transcript kept three ways with the clocks a subtitle asks for, and the refusals:
no sound, too long, outside the roots."""

from datetime import timedelta
from typing import TYPE_CHECKING

import anyio
import pytest

from e_organon.models.audio import TranscribeIn
from e_organon.models.error import AudioTooLongError, NoAudioError, OutsideRootsError
from e_organon.models.flow import Done, Failed, Step
from e_organon.ports.media import MediaInfo
from e_organon.ports.transcribe import Segment
from e_organon.services.audio.transcribe import STEPS, TranscribeAudioService
from support.media import FakeMedia
from support.transcribe import FakeTranscribe, transcript

if TYPE_CHECKING:
    from pathlib import Path

    from e_organon.engines.artifact import ArtifactEngine
    from e_organon.services.files import AllowedFiles

SEGMENTS = (
    Segment(start=0.5, end=1.2, speaker="S01", text="Hola a todos"),
    Segment(start=2.0, end=61.25, speaker="S02", text="el pipeline está listo"),
    Segment(start=62.0, end=63.0, speaker="", text="una voz suelta"),
)


@pytest.fixture
def engine() -> FakeTranscribe:
    return FakeTranscribe(transcript(SEGMENTS))


@pytest.fixture
def media() -> FakeMedia:
    return FakeMedia(MediaInfo(duration=64.0, has_audio=True), chunks=(b"RIFF", b"wave-bytes"))


@pytest.fixture
def tool(
    media: FakeMedia, engine: FakeTranscribe, files: AllowedFiles, artifacts: ArtifactEngine
) -> TranscribeAudioService:
    return TranscribeAudioService(engine, media, files, artifacts, limit=timedelta(minutes=90))


async def test_transcribe_the_five_steps_then_the_answer(tool: TranscribeAudioService, vault: Path) -> None:
    events = [event async for event in tool.events(TranscribeIn(path=str(vault / "mitin.mp4")))]
    assert [event.name for event in events if isinstance(event, Step)] == list(STEPS)
    assert isinstance(events[-1], Done)


async def test_transcribe_streams_the_extracted_wav_to_the_engine(
    tool: TranscribeAudioService, vault: Path, engine: FakeTranscribe, media: FakeMedia
) -> None:
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4"), language="es", hotwords=("pipeline",)))
    ((name, language, hotwords, audio),) = engine.seen
    assert (name, language, hotwords, audio) == ("mitin", "es", ("pipeline",), b"RIFFwave-bytes")
    assert (media.given, out.language, out.duration) == (2, "es", 64.0)


async def test_transcribe_keeps_markdown_json_and_optional_srt(tool: TranscribeAudioService, vault: Path) -> None:
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4"), srt=True))
    markdown, payload, cues = [await anyio.Path(each).read_bytes() for each in (out.path, out.json_path, out.srt_path)]
    assert b"# mitin" in markdown
    assert b"**[S01]** Hola a todos" in markdown
    assert b"`00:01:02.000`" in markdown
    assert b"- duration: 00:01:04.000" in markdown
    assert b'"speaker":"S01"' in payload
    assert b"una voz suelta" in payload
    assert b"1\n00:00:00,500 --> 00:00:01,200\n[S01] Hola a todos" in cues
    assert b"[S02] el pipeline" in cues
    assert b"\nuna voz suelta\n" in cues


async def test_transcribe_without_srt_keeps_two(tool: TranscribeAudioService, vault: Path) -> None:
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    kept = (await anyio.Path(out.path).is_file(), await anyio.Path(out.json_path).is_file())
    assert (out.srt_path, kept) == ("", (True, True))


@pytest.mark.parametrize(
    ("info", "kind"),
    [
        (MediaInfo(duration=64.0, has_audio=False), NoAudioError),
        (MediaInfo(duration=5_401.0, has_audio=True), AudioTooLongError),
    ],
    ids=["no-sound", "too-long"],
)
async def test_transcribe_refuses_before_decoding(
    engine: FakeTranscribe,
    files: AllowedFiles,
    artifacts: ArtifactEngine,
    vault: Path,
    info: MediaInfo,
    kind: type[Exception],
) -> None:
    media = FakeMedia(info)
    tight = TranscribeAudioService(engine, media, files, artifacts, limit=timedelta(minutes=90))
    events = [event async for event in tight.events(TranscribeIn(path=str(vault / "mitin.mp4")))]
    assert isinstance(events[-1], Failed)
    assert events[-1].kind == kind.__name__
    assert (media.given, engine.seen) == (0, [])


async def test_transcribe_a_path_outside_the_roots_fails_the_run(tool: TranscribeAudioService) -> None:
    events = [event async for event in tool.events(TranscribeIn(path="/etc/hostname"))]
    assert isinstance(events[-1], Failed)
    assert events[-1].kind == OutsideRootsError.__name__
