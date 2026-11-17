"""tests/support/transcribe: a speech engine of the tests' own — answers a canned transcript and records every ask,
consuming the audio it is handed so the streaming contract (the wav really arrives) is part of the evidence."""

from typing import TYPE_CHECKING, ClassVar, Self

from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload
from e_organon.ports.transcribe import Transcript

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from e_organon.ports.transcribe import Segment


class FakeTranscribe:
    """The same transcript every time; `seen` records each call as (name, language, hotwords, audio bytes read)."""

    spec: ClassVar[ProviderSpec] = ProviderSpec("fake-transcribe", Locality.SIDECAR, Workload.GPU_BOUND, Isolation.LOOP)

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.seen: list[tuple[str, str, tuple[str, ...], bytes]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def transcribe(
        self, audio: AsyncIterator[bytes], *, name: str, language: str = "", hotwords: tuple[str, ...] = ()
    ) -> Transcript:
        heard = b"".join([chunk async for chunk in audio])
        self.seen.append((name, language, hotwords, heard))
        return self.transcript


def transcript(
    segments: tuple[Segment, ...], *, language: str = "es", text: str = "", seconds: float = 1.5
) -> Transcript:
    """The canned answer the fake gives: joined text unless another is written."""
    return Transcript(
        text=text or "\n".join(each.text for each in segments), segments=segments, language=language, seconds=seconds
    )
