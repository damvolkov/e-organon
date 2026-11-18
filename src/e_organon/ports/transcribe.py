"""ports/transcribe: speech to text — audio in, a timestamped, speaker-aware transcript out."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import msgspec

from e_organon.ports.provider import Provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


##### VALUES #####
class Segment(msgspec.Struct, frozen=True, kw_only=True):
    """One utterance: its span in seconds, its anonymous speaker label — empty when the engine does not tell voices
    apart —, its text, and the speaker's name when something other than the voice told it (empty otherwise)."""

    start: float
    end: float
    speaker: str
    text: str
    name: str = ""


class Transcript(msgspec.Struct, frozen=True, kw_only=True):
    """The whole reading: `text` as the engine returned it (its own markup included), the parsed segments in order, the
    language the engine declared for it (empty where it declares none) and the wall-clock seconds the call took."""

    text: str
    segments: tuple[Segment, ...]
    language: str
    seconds: float


##### PORT #####
@runtime_checkable
class TranscribePort(Provider, Protocol):
    """Transcribes one stream of wav audio (16 kHz, mono): `name` names the recording to the engine, `language` hints a
    code (empty lets the engine decide), `hotwords` steers the spellings the engine would otherwise guess."""

    async def transcribe(
        self, audio: AsyncIterator[bytes], *, name: str, language: str = "", hotwords: tuple[str, ...] = ()
    ) -> Transcript: ...
