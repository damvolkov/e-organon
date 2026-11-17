"""providers/adapters/media/ffmpeg: av containers read and decoded by ffmpeg — the house's one media tool.

The binary runs as our child: a probe answers from `ffprobe`'s json, an extraction streams — ffmpeg writes the wav to
its stdout and the bytes leave here as they arrive, so an hour of audio never sits whole in memory. A reader that stops
early leaves the `async with`, and the child leaves with it: anyio's process kills and reaps what is still running.
What ffmpeg refuses — a container ffprobe cannot tell, a decode that fails, a child gone quiet past `stall` while its
stdout should be flowing — is a `MediaError` with the reason off the child's own stderr, the exit checked, never assumed."""

import subprocess
from typing import TYPE_CHECKING, ClassVar, Final, Self

import anyio
import msgspec

from e_organon.infra.errors import MediaError
from e_organon.ports.media import MediaInfo
from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    from pathlib import Path

    from anyio.abc import ByteReceiveStream

SILENCE: Final = "the child gave nothing more"


##### PROBE WIRE #####
class ProbeStream(msgspec.Struct, frozen=True):
    """One stream of the container: what kind of content it carries."""

    codec_type: str = ""


class ProbeFormat(msgspec.Struct, frozen=True):
    """The container's header: its length in seconds, when it tells one."""

    duration: str = ""


class Probe(msgspec.Struct, frozen=True):
    """ffprobe's json, the two entries this adapter reads."""

    format: ProbeFormat
    streams: list[ProbeStream]


##### ADAPTER #####
class FfmpegAdapter:
    """`MediaPort` on ffmpeg and ffprobe: nothing to open, every call its own bounded child."""

    spec: ClassVar[ProviderSpec] = ProviderSpec("ffmpeg", Locality.EMBEDDED, Workload.CPU_BOUND, Isolation.LOOP)

    __slots__ = ("_ffmpeg", "_ffprobe", "_stall", "_timeout")

    def __init__(
        self, *, timeout: timedelta, stall: timedelta, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"
    ) -> None:
        self._timeout, self._stall, self._ffmpeg, self._ffprobe = timeout, stall, ffmpeg, ffprobe

    ##### PRIVATE #####

    async def _probe_run(self, source: Path) -> Probe:
        """One ffprobe over the header: the decoded shape, or the refusal the binary itself voiced."""
        order = [
            self._ffprobe,
            "-v",
            "error",
            "-hide_banner",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source),
        ]
        try:
            with anyio.fail_after(self._timeout.total_seconds()):
                done = await anyio.run_process(order, check=False)
        except TimeoutError as error:
            msg = f"the media could not be read: {SILENCE} — ffprobe said nothing about {source.name}"
            raise MediaError(msg) from error
        match done.returncode:
            case 0:
                try:
                    return msgspec.json.decode(done.stdout, type=Probe)
                except (msgspec.DecodeError, msgspec.ValidationError) as error:
                    msg = "the media could not be read: ffprobe answered outside its json"
                    raise MediaError(msg) from error
            case code:
                reason = done.stderr.decode(errors="replace").strip() or f"ffprobe died ({code})"
                msg = f"the media could not be read: {reason}"
                raise MediaError(msg)

    @staticmethod
    async def _extract_receive(stream: ByteReceiveStream, *, stall: float) -> bytes:
        """One chunk of the child's stdout; a child gone silent past `stall` is a stuck child, and the end of the
        stream answers as empty — the caller's loop reads it as the stop."""
        try:
            with anyio.fail_after(stall):
                return await stream.receive()
        except TimeoutError as error:
            msg = f"the media could not be decoded: {SILENCE} — ffmpeg went quiet for {stall:g} s"
            raise MediaError(msg) from error
        except anyio.EndOfStream:
            return b""

    @staticmethod
    async def _extract_drain(stream: ByteReceiveStream) -> bytes:
        """What the child wrote to stderr, whole: short by the time it has exited."""
        parts: list[bytes] = []
        try:
            while True:
                parts.append(await stream.receive())
        except anyio.EndOfStream:
            return b"".join(parts)

    ############################################################

    ##### PUBLIC #####

    async def __aenter__(self) -> Self:
        """Nothing to open: every call spawns its own child."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Nothing to release."""

    async def health(self) -> bool:
        """The binaries answer to `version`: the adapter is as alive as the tools it shells out to."""
        try:
            for binary in (self._ffmpeg, self._ffprobe):
                with anyio.fail_after(self._timeout.total_seconds()):
                    done = await anyio.run_process([binary, "-version"], check=False)
                match done.returncode:
                    case 0:
                        continue
                    case _:
                        msg = f"the media tool {binary} refused to answer"
                        raise MediaError(msg)
        except OSError as error:
            msg = f"the media tools are not here: {error}"
            raise MediaError(msg) from error
        return True

    async def probe(self, source: Path) -> MediaInfo:
        """The container's length and whether it carries sound — read from the header, decoding no frame."""
        found = await self._probe_run(source)
        match (found.format.duration, found.streams):
            case ("", _) | (_, []):
                msg = f"the media could not be read: {source.name} has no duration the header tells"
                raise MediaError(msg)
            case duration, streams:
                return MediaInfo(
                    duration=float(duration), has_audio=any(each.codec_type == "audio" for each in streams)
                )

    async def extract(self, source: Path, *, sample_rate: int = 16_000) -> AsyncIterator[bytes]:
        """The container's audio as one mono pcm-s16 wav at `sample_rate`, streamed as ffmpeg decodes it; video dropped."""
        order = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ]
        async with await anyio.open_process(
            order, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as child:
            match (child.stdout, child.stderr):
                case (None, _):
                    msg = "the media could not be decoded: ffmpeg came up with no stdout to read"
                    raise MediaError(msg)
                case (stream, errors):
                    while chunk := await self._extract_receive(stream, stall=self._stall.total_seconds()):
                        yield chunk
                    code = await child.wait()
                    tail = b"" if errors is None else await self._extract_drain(errors)
                    match code:
                        case 0:
                            return
                        case _:
                            reason = tail.decode(errors="replace").strip() or f"ffmpeg died ({code})"
                            msg = f"the media could not be decoded: {source.name}: {reason}"
                            raise MediaError(msg)
