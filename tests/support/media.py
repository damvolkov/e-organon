"""tests/support/media: a media reader of the tests' own — a probe that answers what it was told, an extraction that
streams canned wav bytes and counts the chunks it gave, so a reader that stops early is visible in the record; the
windows it was asked for and the pictures it was asked to sample are kept the same way."""

from typing import TYPE_CHECKING, ClassVar, Self

from e_organon.ports.media import Frame
from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from e_organon.ports.media import MediaInfo


class FakeMedia:
    """One `info` for every file; `extract` yields the canned chunks and records how far the consumer read and the
    `(start, length)` window it was asked; `frames` yields the canned pictures and records what it was asked."""

    spec: ClassVar[ProviderSpec] = ProviderSpec("fake-media", Locality.EMBEDDED, Workload.CPU_BOUND, Isolation.LOOP)

    def __init__(
        self, info: MediaInfo, chunks: tuple[bytes, ...] = (b"RIFFfake-wav",), pictures: tuple[Frame, ...] = ()
    ) -> None:
        self.info = info
        self.chunks = chunks
        self.pictures = pictures
        self.windows: list[tuple[float, float | None]] = []
        self.framed: list[tuple[Path, float, int, int]] = []
        self.probed: list[Path] = []
        self.extracted: list[Path] = []
        self.given: int = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def probe(self, source: Path) -> MediaInfo:
        self.probed.append(source)
        return self.info

    async def extract(
        self, source: Path, *, sample_rate: int = 16_000, start: float = 0.0, length: float | None = None
    ) -> AsyncIterator[bytes]:
        self.extracted.append(source)
        self.windows.append((start, length))
        for chunk in self.chunks:
            self.given += 1
            yield chunk

    async def frames(self, source: Path, *, every: float, width: int, height: int) -> AsyncIterator[Frame]:
        self.framed.append((source, every, width, height))
        for picture in self.pictures:
            yield picture


async def pictures(*at: float) -> AsyncIterator[Frame]:
    """An empty picture at each instant, as a stream."""
    for instant in at:
        yield Frame(at=instant, pixels=b"")
