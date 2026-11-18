"""tests/unit/e_organon/services/audio/transcribe: audio/transcribe over faked media and engine — the steps in order, the
wav really streamed to the engine, a long recording heard in windows (the plan, the shift to the recording's time, the
tail a window dropped asked again), stitched under one set of labels and its speakers named from the video, the
transcript kept three ways with the clocks a subtitle asks for, and the refusals: no sound, too long, outside the roots."""

from datetime import timedelta
from typing import TYPE_CHECKING

import anyio
import pytest

from e_organon.models.audio import TranscribeIn
from e_organon.models.error import AudioTooLongError, NoAudioError, OutsideRootsError
from e_organon.models.flow import Done, Failed, Step
from e_organon.ports.media import Frame, MediaInfo
from e_organon.ports.transcribe import Segment, Transcript
from e_organon.services.audio.transcribe import STEPS, TranscribeAudioService
from support.media import FakeMedia
from support.transcribe import FakeTranscribe, transcript

if TYPE_CHECKING:
    from pathlib import Path

    from e_organon.processors.speakers import SpeakersProcessor
    from support.ocr import FakeOcr
    from support.transcribe import Build

SEGMENTS = (
    Segment(start=0.5, end=1.2, speaker="S01", text="Hola a todos"),
    Segment(start=2.0, end=61.25, speaker="S02", text="el pipeline está listo"),
    Segment(start=62.0, end=63.0, speaker="", text="una voz suelta"),
)


def _say(*spans: tuple[float, float, str, str]) -> Transcript:
    return transcript(tuple(Segment(start=a, end=b, speaker=who, text=said) for a, b, who, said in spans))


@pytest.fixture
def engine() -> FakeTranscribe:
    return FakeTranscribe(transcript(SEGMENTS))


@pytest.fixture
def media() -> FakeMedia:
    return FakeMedia(MediaInfo(duration=64.0, has_audio=True), chunks=(b"RIFF", b"wave-bytes"))


@pytest.fixture
def tool(media: FakeMedia, engine: FakeTranscribe, build: Build) -> TranscribeAudioService:
    return build(engine, media)


async def test_transcribe_the_six_steps_then_the_answer(tool: TranscribeAudioService, vault: Path) -> None:
    events = [event async for event in tool.events(TranscribeIn(path=str(vault / "mitin.mp4")))]
    assert [event.name for event in events if isinstance(event, Step)] == list(STEPS)
    assert (len(STEPS), isinstance(events[-1], Done)) == (6, True)


async def test_transcribe_streams_the_extracted_wav_to_the_engine(
    tool: TranscribeAudioService, vault: Path, engine: FakeTranscribe, media: FakeMedia
) -> None:
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4"), language="es", hotwords=("pipeline",)))
    ((name, language, hotwords, audio),) = engine.seen
    assert (name, language, hotwords, audio) == ("mitin", "es", ("pipeline",), b"RIFFwave-bytes")
    assert (media.given, out.language, out.duration) == (2, "es", 64.0)
    assert (media.windows, out.windows, out.covered) == ([(0.0, 64.0)], 1, 63.0)


async def test_transcribe_keeps_markdown_json_and_optional_srt(tool: TranscribeAudioService, vault: Path) -> None:
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4"), srt=True))
    markdown, payload, cues = [await anyio.Path(each).read_bytes() for each in (out.path, out.json_path, out.srt_path)]
    assert b"# mitin" in markdown
    assert b"**[S01]** Hola a todos" in markdown
    assert b"`00:01:02.000`" in markdown
    assert b"- duration: 00:01:04.000" in markdown
    assert (b"- covered: 00:01:03.000" in markdown, b"- windows: 1\n" in markdown) == (True, True)
    assert b'"speaker":"S01"' in payload
    assert b"una voz suelta" in payload
    assert (b'"covered":63.0' in payload, b'"windows":1' in payload, b'"name":""' in payload) == (True, True, True)
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
    engine: FakeTranscribe, build: Build, vault: Path, info: MediaInfo, kind: type[Exception]
) -> None:
    media = FakeMedia(info)
    events = [event async for event in build(engine, media).events(TranscribeIn(path=str(vault / "mitin.mp4")))]
    assert isinstance(events[-1], Failed)
    assert events[-1].kind == kind.__name__
    assert (media.given, engine.seen) == (0, [])


async def test_transcribe_a_path_outside_the_roots_fails_the_run(tool: TranscribeAudioService) -> None:
    events = [event async for event in tool.events(TranscribeIn(path="/etc/hostname"))]
    assert isinstance(events[-1], Failed)
    assert events[-1].kind == OutsideRootsError.__name__


##### WINDOWS #####


@pytest.mark.parametrize(
    ("duration", "asked"),
    [
        (64.0, [(0.0, 64.0)]),
        (100.0, [(0.0, 100.0)]),
        (105.0, [(0.0, 100.0), (90.0, 15.0)]),
        (185.0, [(0.0, 100.0), (90.0, 95.0)]),
        (190.0, [(0.0, 100.0), (90.0, 100.0)]),
        (250.0, [(0.0, 100.0), (90.0, 100.0), (180.0, 70.0)]),
    ],
    ids=["short", "fits", "tail-past-overlap", "no-start-in-the-last-overlap", "start-at-the-last-overlap", "three"],
)
async def test_transcribe_plans_windows_by_step(
    build: Build, vault: Path, duration: float, asked: list[tuple[float, float]]
) -> None:
    media = FakeMedia(MediaInfo(duration=duration, has_audio=True))
    tool = build(FakeTranscribe(transcript(())), media, window=timedelta(seconds=100), overlap=timedelta(seconds=10))
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    assert (media.windows, out.windows) == (asked, len(asked))


async def test_transcribe_shifts_each_window_to_the_recordings_time_and_stitches_the_labels(
    build: Build, vault: Path
) -> None:
    engine = FakeTranscribe(
        _say((0, 50, "S01", "a"), (50, 99, "S02", "b")),
        _say((0, 9, "S01", "b again"), (10, 60, "S01", "c")),
    )
    media = FakeMedia(MediaInfo(duration=150.0, has_audio=True))
    tool = build(engine, media, window=timedelta(seconds=100), overlap=timedelta(seconds=10))
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    ### the second window's S01 spoke the overlap alongside the first's S02: one voice; the cut at 94.5 keeps "b" once.
    assert [(each.start, each.end, each.speaker, each.text) for each in out.segments] == [
        (0, 50, "S01", "a"),
        (50, 99, "S02", "b"),
        (100, 150, "S02", "c"),
    ]
    assert (out.windows, out.covered) == (2, 150.0)


def _stopping_short() -> FakeTranscribe:
    return FakeTranscribe(
        _say((0, 50, "S01", "uno")),
        _say((0, 5, "S01", "dos"), (8, 30, "S01", "dos b")),
        _say((0, 4, "S01", "tres"), (6, 40, "S01", "tres b")),
    )


@pytest.mark.parametrize(
    ("retries", "asked"),
    [
        (0, [(0.0, 100.0)]),
        (1, [(0.0, 100.0), (40.0, 60.0)]),
        (2, [(0.0, 100.0), (40.0, 60.0), (60.0, 40.0)]),
        (5, [(0.0, 100.0), (40.0, 60.0), (60.0, 40.0)]),
    ],
    ids=["none", "one", "two", "stops-when-heard"],
)
async def test_transcribe_asks_again_from_overlap_before_where_a_window_stopped(
    build: Build, vault: Path, retries: int, asked: list[tuple[float, float]]
) -> None:
    media = FakeMedia(MediaInfo(duration=100.0, has_audio=True))
    tool = build(
        _stopping_short(), media, window=timedelta(seconds=100), overlap=timedelta(seconds=10), retries=retries
    )
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    assert (media.windows, out.windows) == (asked, len(asked))


async def test_transcribe_a_re_asked_window_is_stitched_over_what_was_heard(build: Build, vault: Path) -> None:
    media = FakeMedia(MediaInfo(duration=100.0, has_audio=True))
    tool = build(_stopping_short(), media, window=timedelta(seconds=100), overlap=timedelta(seconds=10))
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    assert [(each.speaker, each.text) for each in out.segments] == [("S01", "uno"), ("S01", "dos b"), ("S01", "tres b")]
    assert out.covered == 100.0


@pytest.mark.parametrize(
    ("languages", "expected"),
    [(("", "en", "es"), "en"), (("", "", ""), ""), (("es", "en", "en"), "es")],
    ids=["first-told", "none-told", "first-wins"],
)
async def test_transcribe_the_language_is_the_first_a_window_told(
    build: Build, vault: Path, languages: tuple[str, str, str], expected: str
) -> None:
    first, second, third = (
        transcript((Segment(start=0, end=end, speaker="S01", text="x"),), language=language)
        for end, language in zip((50, 30, 40), languages, strict=True)
    )
    media = FakeMedia(MediaInfo(duration=100.0, has_audio=True))
    tool = build(
        FakeTranscribe(first, second, third), media, window=timedelta(seconds=100), overlap=timedelta(seconds=10)
    )
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    assert (out.language, out.windows) == (expected, 3)


async def test_transcribe_an_empty_answer_is_the_silence_it_heard_and_is_kept(build: Build, vault: Path) -> None:
    engine = FakeTranscribe(transcript(()))
    media = FakeMedia(MediaInfo(duration=100.0, has_audio=True))
    tool = build(engine, media, window=timedelta(seconds=100), overlap=timedelta(seconds=10))
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4")))
    assert (len(engine.seen), out.segments, out.covered, out.speakers) == (1, [], 0.0, {})


##### NAMES #####


@pytest.fixture
def talk() -> FakeTranscribe:
    return FakeTranscribe(_say((0.5, 3.5, "S01", "Hola"), (5.0, 6.0, "S02", "Adiós")))


async def test_transcribe_names_each_speaker_from_the_tile_the_video_rings(
    build: Build,
    vault: Path,
    talk: FakeTranscribe,
    filmed: FakeMedia,
    speakers: SpeakersProcessor,
    ocr: FakeOcr,
) -> None:
    tool = build(talk, filmed, speakers=speakers, frames=(timedelta(seconds=2), 64, 36))
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4"), srt=True))
    assert out.speakers == {"S01": "Ana García"}
    assert [(each.speaker, each.name) for each in out.segments] == [("S01", "Ana García"), ("S02", "")]
    assert [framed[1:] for framed in filmed.framed] == [(2.0, 64, 36)]
    ### one strip, one place, looking alike three times: read once.
    assert len(ocr.seen) == 1
    markdown, payload, cues = [
        (await anyio.Path(each).read_bytes()).decode() for each in (out.path, out.json_path, out.srt_path)
    ]
    assert ("**[Ana García]** Hola" in markdown, "**[S02]** Adiós" in markdown) == (True, True)
    assert '"name":"Ana García"' in payload
    assert ("[Ana García] Hola" in cues, "[S02] Adiós" in cues) == (True, True)


@pytest.mark.parametrize(
    ("video", "names", "speaking", "framing"),
    [(False, True, True, True), (True, False, True, True), (True, True, False, True), (True, True, True, False)],
    ids=["no-video", "names-off", "no-processor", "no-frames-config"],
)
async def test_transcribe_names_nobody_unless_video_names_processor_and_frames_are_all_there(
    build: Build,
    vault: Path,
    talk: FakeTranscribe,
    speakers: SpeakersProcessor,
    video: bool,
    names: bool,
    speaking: bool,
    framing: bool,
) -> None:
    pictures = tuple(Frame(at=at, pixels=b"") for at in (1.0, 2.0, 3.0))
    media = FakeMedia(MediaInfo(duration=10.0, has_audio=True, has_video=video), pictures=pictures)
    tool = build(
        talk,
        media,
        speakers=speakers if speaking else None,
        frames=(timedelta(seconds=2), 64, 36) if framing else None,
    )
    out = await tool.run(TranscribeIn(path=str(vault / "mitin.mp4"), names=names))
    assert (media.framed, out.speakers, {each.name for each in out.segments}) == ([], {}, {""})
