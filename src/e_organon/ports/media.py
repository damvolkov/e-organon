"""ports/media: what an av container holds — its shape read, its audio taken out."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import msgspec

from e_organon.ports.provider import Provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


##### VALUES #####
class MediaInfo(msgspec.Struct, frozen=True, kw_only=True):
    """What a container is: its length in seconds and whether it carries any sound at all."""

    duration: float
    has_audio: bool


##### PORT #####
@runtime_checkable
class MediaPort(Provider, Protocol):
    """Reads and decodes av containers. `probe` answers from the header, decoding no frame; `extract` streams the
    container's audio as one mono wav at `sample_rate` — the shape every speech engine reads alike, video dropped."""

    async def probe(self, source: Path) -> MediaInfo: ...

    def extract(self, source: Path, *, sample_rate: int = 16_000) -> AsyncIterator[bytes]: ...
