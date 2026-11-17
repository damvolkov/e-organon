"""providers/drivers/speech/moss: long-form speech to text on a MOSS-Transcribe-Diarize server.

The service, not the library: the model (0.9B, ~2 GB) lives in its own container — sglang-omni or vLLM, both speaking
the OpenAI audio endpoint — so the app carries no torch. One request streams: the wav is posted as multipart form-data
fed from the media adapter's generator (never held whole), and the server answers `verbose_json` — the transcript
already split into timestamped, speaker-labelled segments. Temperature 0: transcription is reading, not invention;
`max_new_tokens` is raised past the default because one diarized hour outgrows it. The whole exchange is bounded by
`deadline`, carried as the request's own timeout. It speaks through an `HTTPPort` received by constructor, absolute
URLs on the `url` it is given.

Hotwords ride the `prompt` field, and the field *replaces* the server's built-in instruction — so the driver composes
what the card prescribes: the trained prompt with the `HOTWORDS` hint appended, never the hint alone. What the server
answers outside its contract is a `TranscribeError` naming the shape it did answer — never a guess at what it meant.

The recording travels whole and the server never chunks it for MOSS (`allow_audio_chunking` is False for it — 128 K
context in one pass is its point). The 4090 bench measured what that costs the host: the encoder materializes the
audio (~1.4 GiB per gelu pass over a 61-min recording) and the KV holds audio plus transcript to the last token
(~20 k tokens per dense minute). The card is the sidecar's alone while it serves (`docs/decisions.md`).

The slot that bounds how many recordings transcribe at once is the queue's (`ORGANON_SLOTS_AUDIO`), so the driver
gates nothing itself: one GPU, one honest limit, named once."""

import re
import time
from collections.abc import Mapping
from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar, Final, Self
from uuid import uuid4

from e_organon.infra.errors import TranscribeError
from e_organon.infra.logger import logger
from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload
from e_organon.ports.transcribe import Segment, Transcript

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from datetime import timedelta

    from pydantic import HttpUrl

    from e_organon.ports.http import HTTPPort


##### BANK #####
class MossField(StrEnum):
    """The OpenAI audio endpoint's own form names, and the one answer format this driver reads."""

    FILE = auto()
    MODEL = auto()
    RESPONSE_FORMAT = auto()
    TEMPERATURE = auto()
    MAX_NEW_TOKENS = auto()
    LANGUAGE = auto()
    PROMPT = auto()


VERBOSE: Final = "verbose_json"
### sglang-omni parses its own markup into segments whose text keeps the speaker as this prefix — never a field; a
### crowded room is labelled beyond two digits (the server's own regex is `S\d{2,}`).
SPEAKER_PREFIX: Final = re.compile(r"^\[(S\d{2,})\]")
### the instruction the model was trained on, verbatim from its card: `prompt` replaces it entirely, so a hotword hint
### rides only as the card prescribes — this text with the hint appended. Never the hint alone.
PROMPT_BASE: Final = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"  # noqa: RUF001
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"  # noqa: RUF001
)
HOTWORDS: Final = "热词提示："  # noqa: RUF001
HOTWORD_JOIN: Final = ", "
### the endpoint's per-field values this driver fixes: reading, not invention.
ZERO: Final = "0"
### the server echoes the requested language back, or this when none was asked: a request mirror, never a detection.
AUTO: Final = "auto"
UNSAFE: Final = frozenset("\"'\\\r\n")
### a streamed body carries no filename to read a type from: the multipart header, with the boundary the body
### actually separates on, is what makes it a form at all — without it the server answers 422.
MULTIPART: Final = "multipart/form-data; boundary={}"
CONTENT_TYPE: Final = "Content-Type"


##### DRIVER #####
class MossDriver:
    """A MOSS-Transcribe-Diarize server behind `TranscribePort`, one streaming request per recording."""

    spec: ClassVar[ProviderSpec] = ProviderSpec(
        "sglang-omni",
        Locality.SIDECAR,
        Workload.GPU_BOUND,
        Isolation.LOOP,
        fork_safe=True,
        ### the whole sidecar container, not the weights: `--mem-fraction-static 0.75` of the 24 GB card.
        memory_hint_mb=18_432,
    )

    __slots__ = ("_deadline", "_http", "_max_new_tokens", "_model", "_url")

    def __init__(
        self,
        http: HTTPPort,
        *,
        url: HttpUrl | None,
        model: str,
        deadline: timedelta,
        max_new_tokens: int,
    ) -> None:
        self._http = http
        self._model, self._deadline, self._max_new_tokens = model, deadline, max_new_tokens
        match url:
            case None:
                msg = "MOSS_URL is unset: there is no speech sidecar to speak to"
                raise TranscribeError(msg)
            case _:
                self._url = str(url).rstrip("/")

    ##### PRIVATE #####

    @staticmethod
    def _multipart_field(name: str, value: str, boundary: str) -> bytes:
        """One form-data part of a plain field."""
        return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()

    def _multipart_fields(self, language: str, hotwords: tuple[str, ...], boundary: str) -> list[bytes]:
        """The request's plain fields, the server's own names: the whole reading asked for and nothing tuned by hand
        beyond it — bar the token ceiling, which an hour of diarized speech needs raised."""
        fields = [
            self._multipart_field(MossField.MODEL, self._model, boundary),
            self._multipart_field(MossField.RESPONSE_FORMAT, VERBOSE, boundary),
            self._multipart_field(MossField.TEMPERATURE, ZERO, boundary),
            self._multipart_field(MossField.MAX_NEW_TOKENS, str(self._max_new_tokens), boundary),
        ]
        more: Mapping[str, str] = {
            **({MossField.LANGUAGE: language} if language else {}),
            **({MossField.PROMPT: PROMPT_BASE + HOTWORDS + HOTWORD_JOIN.join(hotwords)} if hotwords else {}),
        }
        return [*fields, *(self._multipart_field(key, value, boundary) for key, value in more.items())]

    async def _multipart_body(
        self, audio: AsyncIterator[bytes], name: str, language: str, hotwords: tuple[str, ...], boundary: str
    ) -> AsyncGenerator[bytes]:
        """The whole multipart body as it is produced: fields, the wav streamed from its decoder, the closing bound."""
        safe = "".join(each for each in name if each not in UNSAFE)
        for part in self._multipart_fields(language, hotwords, boundary):
            yield part
        yield (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{MossField.FILE}"; filename="{safe}.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        async for chunk in audio:
            yield chunk
        yield f"\r\n--{boundary}--\r\n".encode()

    @staticmethod
    def _answer_segment(each: object) -> Segment:
        """One verbose_json segment: its span, its anonymous speaker and its text. sglang-omni carries the speaker as
        the `[Sxx]` prefix of the text; a server with its own field (vLLM's dialect) is read from that, an index being
        0-based while the label the room reads is 1-based. No prefix, no field: one voice heard, none to name."""
        match each:
            case {"start": int() | float() as start, "end": int() | float() as end, "text": str() as raw, **rest}:
                match rest.get("speaker", rest.get("speaker_id", "")):
                    case str() as label if label:
                        return Segment(start=float(start), end=float(end), speaker=label, text=raw)
                    case int() as index:
                        return Segment(start=float(start), end=float(end), speaker=f"S{index + 1:02d}", text=raw)
                    case _:
                        match SPEAKER_PREFIX.match(raw):
                            case None:
                                return Segment(start=float(start), end=float(end), speaker="", text=raw)
                            case found:
                                return Segment(
                                    start=float(start),
                                    end=float(end),
                                    speaker=found[1],
                                    text=raw[found.end() :].lstrip(),
                                )
            case _:
                msg = f"a segment outside the contract: {str(each)[:160]}"
                raise TranscribeError(msg)

    @staticmethod
    def _answer_transcript(body: object) -> tuple[str, tuple[Segment, ...], str]:
        """The verbose_json answer as its parts: the text, the segments, the language declared — or the shape it missed."""
        match body:
            case {"text": str() as text, "segments": list() as raw, **rest}:
                language = str(rest.get("language") or "")
                ### 'auto' is the echo of an unhinted request, not a heard language: the port says nothing over it.
                return (
                    text,
                    tuple(MossDriver._answer_segment(each) for each in raw),
                    "" if language == AUTO else language,
                )
            case Mapping():
                msg = f"the server answered outside verbose_json (keys: {sorted(body)})"
                raise TranscribeError(msg)
            case _:
                msg = f"the server answered not an object at all: {body!r:.160}"
                raise TranscribeError(msg)

    ############################################################

    ##### PUBLIC #####

    async def __aenter__(self) -> Self:
        """Nothing to open: the HTTP pool is the lifespan's, the model is the server's."""
        logger.info("moss opened", url=self._url, model=self._model)
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Nothing to release."""

    async def health(self) -> bool:
        """The served models, as any OpenAI-compatible server lists them (`/models`): the sidecar answers and holds the model."""
        await self._http.get(f"{self._url}/models", within=self._deadline)
        return True

    async def transcribe(
        self, audio: AsyncIterator[bytes], *, name: str, language: str = "", hotwords: tuple[str, ...] = ()
    ) -> Transcript:
        """The recording transcribed: the wav streamed up, the answered transcript measured and typed."""
        started = time.monotonic()
        boundary = uuid4().hex
        body = await self._http.upload(
            f"{self._url}/audio/transcriptions",
            self._multipart_body(audio, name, language, hotwords, boundary),
            within=self._deadline,
            headers={CONTENT_TYPE: MULTIPART.format(boundary)},
        )
        text, segments, heard = self._answer_transcript(body)
        seconds = time.monotonic() - started
        logger.info("moss transcribed", name=name, segments=len(segments), seconds=round(seconds, 1))
        return Transcript(text=text, segments=segments, language=heard, seconds=seconds)
