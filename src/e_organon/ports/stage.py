"""ports/stage: who holds the floor in a meeting's video — the tiles a call app marks as speaking, read off one picture."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import msgspec

from e_organon.ports.provider import Provider

if TYPE_CHECKING:
    from e_organon.ports.media import Frame


##### VALUES #####
class Ring(msgspec.Struct, frozen=True, kw_only=True):
    """One tile marked as speaking: its place `(left, top, width, height)` in the picture's pixels, the strip at its foot
    where the app writes the person's name (a PNG, for an OCR to read), and that strip's coarse look — grey bytes small
    enough to compare two pictures of the same strip without reading either."""

    place: tuple[int, int, int, int]
    strip: bytes
    look: bytes


##### PORT #####
@runtime_checkable
class StagePort(Provider, Protocol):
    """`rings` the tiles marked as speaking in one `bgr24` picture of `width` by `height`: none when nobody is, or the
    picture is no grid (a shared screen); several when the app marks more than one voice at once."""

    async def rings(self, frame: Frame, *, width: int, height: int) -> tuple[Ring, ...]: ...
