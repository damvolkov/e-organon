"""e_organon.services.audio.transcribe: audio/transcribe — a recording, whole, into a timestamped speaker-aware transcript.

A video is read for what it carries of sound: its audio extracted as one mono wav, streamed to the speech engine while
ffmpeg still decodes it — the hour of audio never sits whole anywhere. The engine answers segments; this tool renders
them three ways and keeps them in the outbox: Markdown for a reader to skim, JSON for a program to hold, SRT when the
caller wants subtitles. The recording's length is read from the header and held to the limit before a frame is decoded:
an engine built for one pass of the longest recordings is not asked to swallow the impossible.

Five steps: the file resolved, its shape probed, its speech transcribed, the transcript rendered, the files kept."""

from typing import TYPE_CHECKING

import msgspec

from e_organon.models.audio import SegmentOut, TranscribeIn, TranscribeOut
from e_organon.models.error import AudioTooLongError, NoAudioError
from e_organon.models.flow import Done, Event, Step
from e_organon.services.flow import ToolFlow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta

    from e_organon.engines.artifact import ArtifactEngine
    from e_organon.ports.media import MediaPort
    from e_organon.ports.transcribe import Segment, TranscribePort, Transcript
    from e_organon.services.files import AllowedFiles

STEPS: tuple[str, ...] = ("resolve", "probe", "transcribe", "render", "keep")


##### RENDER #####
class Render:
    """The transcript as each of its three readings: Markdown to skim, JSON to hold, SRT to burn."""

    __slots__ = ("_duration", "_stem", "_transcript")

    def __init__(self, stem: str, duration: float, transcript: Transcript) -> None:
        self._stem, self._duration, self._transcript = stem, duration, transcript

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
    def _common_speaker(segment: Segment) -> str:
        """What a line leads with when the engine told voices apart: its speaker label."""
        return f"**[{segment.speaker}]** " if segment.speaker else ""

    def _markdown_line(self, segment: Segment) -> str:
        return f"`{self._common_clock(segment.start)}` {self._common_speaker(segment)}{segment.text}"

    def _srt_cue(self, index: int, segment: Segment) -> str:
        said = f"[{segment.speaker}] {segment.text}" if segment.speaker else segment.text
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
            "segments": [msgspec.structs.asdict(each) for each in self._transcript.segments],
        }
        return msgspec.json.encode(payload)

    def srt(self) -> bytes:
        """The transcript as SubRip subtitles: each utterance one cue."""
        cues = (self._srt_cue(index, each) for index, each in enumerate(self._transcript.segments, start=1))
        return "\n".join(cues).encode()


##### TOOL #####
class TranscribeAudioService(ToolFlow[TranscribeIn, TranscribeOut]):
    """audio/transcribe: a recording transcribed, its transcript kept three ways."""

    __slots__ = ("_artifacts", "_files", "_limit", "_media", "_moss")

    def __init__(
        self,
        moss: TranscribePort,
        media: MediaPort,
        files: AllowedFiles,
        artifacts: ArtifactEngine,
        *,
        limit: timedelta,
    ) -> None:
        self._moss, self._media, self._files, self._artifacts, self._limit = moss, media, files, artifacts, limit

    ##### PRIVATE #####

    def _events_bound(self, name: str, duration: float) -> None:
        """The recording too long for one pass is refused on the header's word, nothing decoded."""
        match duration <= self._limit.total_seconds():
            case False:
                msg = f"{name} runs {duration:.0f} s; the limit is {self._limit}"
                raise AudioTooLongError(msg)

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
            case True:
                self._events_bound(source.name, info.duration)
        yield Step(name=STEPS[2], index=2, total=len(STEPS))
        transcript = await self._moss.transcribe(
            self._media.extract(source), name=source.stem, language=data.language, hotwords=data.hotwords
        )
        yield Step(name=STEPS[3], index=3, total=len(STEPS))
        rendered = Render(source.stem, info.duration, transcript)
        markdown, payload = rendered.markdown(), rendered.json()
        subtitles = rendered.srt() if data.srt else b""
        yield Step(name=STEPS[4], index=4, total=len(STEPS))
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
                segments=[SegmentOut(**msgspec.structs.asdict(each)) for each in transcript.segments],
            )
        )
