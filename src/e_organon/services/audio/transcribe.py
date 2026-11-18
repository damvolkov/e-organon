"""e_organon.services.audio.transcribe: audio/transcribe — a recording, whole, into a timestamped speaker-aware transcript,
its speakers named when the video shows who speaks.

A video is read for what it carries of sound: its audio extracted as mono wav, streamed to the speech engine while
ffmpeg still decodes it — the hour of audio never sits whole anywhere. A recording longer than `window` is heard in
windows of it that overlap by `overlap`, one request each: a single pass of an hour was measured to stop at 37
minutes, the engine answering success. A window whose answer stops hearing more than `overlap` before its end is asked
again for the rest, from `overlap` before where it stopped, up to `retries` times; an answer with nothing in it is the
silence it heard, and is kept. With a video and a stage reader, the pictures are looked at for the speaking tile, and
each window's labels take the names the video showed while they spoke — each window on its own, before stitching: the
names then decide who is who across windows (`processors/speakers`). The transcript is rendered three ways and kept in the outbox: Markdown for a reader to skim, JSON for a
program to hold, SRT when the caller wants subtitles. The recording's length is read from the header and held to the
limit before a frame is decoded.

Six steps: the file resolved, its shape probed, its speech transcribed, its speakers named, the transcript rendered,
the files kept."""

from typing import TYPE_CHECKING

import msgspec

from e_organon.models.audio import SegmentOut, TranscribeIn, TranscribeOut
from e_organon.models.error import AudioTooLongError, NoAudioError
from e_organon.models.flow import Done, Event, Step
from e_organon.processors.speakers import SpeakersProcessor, Window
from e_organon.services.flow import ToolFlow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    from pathlib import Path

    from e_organon.engines.artifact import ArtifactEngine
    from e_organon.ports.media import MediaInfo, MediaPort
    from e_organon.ports.transcribe import Segment, TranscribePort, Transcript
    from e_organon.services.files import AllowedFiles

STEPS: tuple[str, ...] = ("resolve", "probe", "transcribe", "name", "render", "keep")


##### RENDER #####
class Render:
    """The transcript as each of its three readings: Markdown to skim, JSON to hold, SRT to burn."""

    __slots__ = ("_covered", "_duration", "_stem", "_transcript", "_windows")

    def __init__(self, stem: str, duration: float, transcript: Transcript, *, windows: int, covered: float) -> None:
        self._stem, self._duration, self._transcript = stem, duration, transcript
        self._windows, self._covered = windows, covered

    ##### PRIVATE #####

    @staticmethod
    def _common_clock(seconds: float, *, comma: bool = False) -> str:
        """The instant as `hh:mm:ss.mmm` — with the comma the SubRip clock asks for in place of the dot. Whole
        milliseconds from the start: the float stays away from the arithmetic a cue is read by."""
        hours, rest = divmod(round(max(0.0, seconds) * 1000), 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        secs, millis = divmod(rest, 1_000)
        clock = f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        return clock.replace(".", ",") if comma else clock

    @staticmethod
    def _common_who(segment: Segment) -> str:
        """Who said it: the name the video showed, else the engine's label, else nobody."""
        return segment.name or segment.speaker

    @classmethod
    def _common_speaker(cls, segment: Segment) -> str:
        """What a line leads with when the voice is told apart: its speaker."""
        return f"**[{who}]** " if (who := cls._common_who(segment)) else ""

    def _markdown_line(self, segment: Segment) -> str:
        return f"`{self._common_clock(segment.start)}` {self._common_speaker(segment)}{segment.text}"

    def _srt_cue(self, index: int, segment: Segment) -> str:
        said = f"[{who}] {segment.text}" if (who := self._common_who(segment)) else segment.text
        span = f"{self._common_clock(segment.start, comma=True)} --> {self._common_clock(segment.end, comma=True)}"
        return f"{index}\n{span}\n{said}\n"

    ############################################################

    ##### PUBLIC #####

    def markdown(self) -> bytes:
        """The transcript as a document: the reading's shape up top, then who said what from when."""
        heard = self._transcript.language or "unknown"
        lines = [
            f"# {self._stem}",
            "",
            f"- language: {heard}",
            f"- duration: {self._common_clock(self._duration)}",
            f"- covered: {self._common_clock(self._covered)}",
            f"- windows: {self._windows}",
            f"- segments: {len(self._transcript.segments)}",
            "",
            *(self._markdown_line(each) for each in self._transcript.segments),
            "",
        ]
        return "\n".join(lines).encode()

    def json(self) -> bytes:
        """The transcript whole: its language, its length, its segments — the words apart, the truth kept."""
        payload = {
            "language": self._transcript.language,
            "duration": self._duration,
            "covered": self._covered,
            "windows": self._windows,
            "segments": [msgspec.structs.asdict(each) for each in self._transcript.segments],
        }
        return msgspec.json.encode(payload)

    def srt(self) -> bytes:
        """The transcript as SubRip subtitles: each utterance one cue."""
        cues = (self._srt_cue(index, each) for index, each in enumerate(self._transcript.segments, start=1))
        return "\n".join(cues).encode()


##### TOOL #####
class TranscribeAudioService(ToolFlow[TranscribeIn, TranscribeOut]):
    """audio/transcribe: a recording transcribed in windows, its speakers named, its transcript kept three ways."""

    __slots__ = (
        "_artifacts",
        "_files",
        "_frames",
        "_limit",
        "_media",
        "_moss",
        "_overlap",
        "_retries",
        "_speakers",
        "_window",
    )

    def __init__(
        self,
        moss: TranscribePort,
        media: MediaPort,
        files: AllowedFiles,
        artifacts: ArtifactEngine,
        *,
        limit: timedelta,
        window: timedelta,
        overlap: timedelta,
        retries: int,
        speakers: SpeakersProcessor | None = None,
        frames: tuple[timedelta, int, int] | None = None,
    ) -> None:
        self._moss, self._media, self._files, self._artifacts, self._limit = moss, media, files, artifacts, limit
        self._window, self._overlap, self._retries = window.total_seconds(), overlap.total_seconds(), retries
        self._speakers, self._frames = speakers, frames

    ##### PRIVATE #####

    def _events_bound(self, name: str, duration: float) -> None:
        """The recording too long for the tool is refused on the header's word, nothing decoded."""
        match duration <= self._limit.total_seconds():
            case False:
                msg = f"{name} runs {duration:.0f} s; the limit is {self._limit}"
                raise AudioTooLongError(msg)

    def _hear_plan(self, duration: float) -> list[tuple[float, float]]:
        """The windows a recording is heard in: one when it fits a window, else steps of `window - overlap`."""
        match duration <= self._window:
            case True:
                return [(0.0, duration)]
            case _:
                step = self._window - self._overlap
                starts = [index * step for index in range(int((duration - self._overlap) // step) + 1)]
                return [
                    (start, min(start + self._window, duration)) for start in starts if start < duration - self._overlap
                ]

    async def _hear_one(self, source: Path, start: float, end: float, data: TranscribeIn) -> Transcript:
        """One window asked of the engine, its segments moved to the recording's time."""
        heard = await self._moss.transcribe(
            self._media.extract(source, start=start, length=end - start),
            name=source.stem,
            language=data.language,
            hotwords=data.hotwords,
        )
        shifted = tuple(
            msgspec.structs.replace(each, start=each.start + start, end=each.end + start) for each in heard.segments
        )
        return msgspec.structs.replace(heard, segments=shifted)

    async def _hear(self, source: Path, info: MediaInfo, data: TranscribeIn) -> tuple[list[Window], Transcript]:
        """Every window heard, and a window that stopped short asked again for the rest; the last answer's shape (its
        language, its time) summed over all."""
        pending = [(start, end, 0) for start, end in self._hear_plan(info.duration)]
        windows: list[Window] = []
        answers: list[Transcript] = []
        while pending:
            start, end, tried = pending.pop(0)
            answer = await self._hear_one(source, start, end, data)
            window = Window(start=start, end=end, segments=answer.segments)
            windows.append(window)
            answers.append(answer)
            stopped = window.heard
            match (bool(answer.segments), stopped < end - self._overlap, tried < self._retries):
                case (True, True, True):
                    pending.insert(0, (max(start, stopped - self._overlap), end, tried + 1))
        language = next((each.language for each in answers if each.language), "")
        whole = msgspec.structs.replace(
            answers[0],
            text="\n".join(each.text for each in answers),
            language=language,
            seconds=sum(each.seconds for each in answers),
        )
        return windows, whole

    async def _name(self, source: Path, info: MediaInfo, data: TranscribeIn, windows: list[Window]) -> list[Window]:
        """Each window's speakers named from the video's pictures on its own, before any stitching — when there is a
        video, a reader of its stage, and names asked."""
        match (self._speakers, self._frames, info.has_video and data.names):
            case (SpeakersProcessor() as speakers, (every, width, height), True):
                seen = await speakers.heard(
                    self._media.frames(source, every=every.total_seconds(), width=width, height=height)
                )
                return [
                    Window(start=each.start, end=each.end, segments=speakers.name(each.segments, seen))
                    for each in windows
                ]
            case _:
                return windows

    ############################################################

    async def _events_run(self, data: TranscribeIn) -> AsyncIterator[Event]:
        yield Step(name=STEPS[0], index=0, total=len(STEPS))
        source = await self._files.resolve(data.path)
        yield Step(name=STEPS[1], index=1, total=len(STEPS))
        info = await self._media.probe(source)
        match info.has_audio:
            case False:
                msg = f"{source.name} carries no sound"
                raise NoAudioError(msg)
            case _:
                self._events_bound(source.name, info.duration)
        yield Step(name=STEPS[2], index=2, total=len(STEPS))
        windows, heard = await self._hear(source, info, data)
        yield Step(name=STEPS[3], index=3, total=len(STEPS))
        named = SpeakersProcessor.stitch(await self._name(source, info, data, windows))
        transcript = msgspec.structs.replace(heard, segments=named)
        covered = max((each.end for each in named), default=0.0)
        yield Step(name=STEPS[4], index=4, total=len(STEPS))
        rendered = Render(source.stem, info.duration, transcript, windows=len(windows), covered=covered)
        markdown, payload = rendered.markdown(), rendered.json()
        subtitles = rendered.srt() if data.srt else b""
        yield Step(name=STEPS[5], index=5, total=len(STEPS))
        md = await self._artifacts.keep(markdown, source.stem, ".md")
        js = await self._artifacts.keep(payload, source.stem, ".json")
        cues = await self._artifacts.keep(subtitles, source.stem, ".srt") if data.srt else None
        yield Done(
            result=TranscribeOut(
                path=str(md),
                json_path=str(js),
                srt_path=str(cues) if cues else "",
                language=transcript.language,
                duration=info.duration,
                seconds=transcript.seconds,
                windows=len(windows),
                covered=covered,
                speakers={each.speaker: each.name for each in named if each.name},
                segments=[SegmentOut(**msgspec.structs.asdict(each)) for each in named],
            )
        )
