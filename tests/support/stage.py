"""tests/support/stage: the stage of a call, two ways — `FakeStage`, a reader that answers the rings it was told for each
instant, and `grid`, a picture of the tests' own with Teams' speaking ring drawn where it is asked."""

from typing import TYPE_CHECKING, ClassVar, Final, Self

import cv2
import numpy as np

from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload
from e_organon.ports.stage import Ring

if TYPE_CHECKING:
    from collections.abc import Mapping

    from e_organon.ports.media import Frame

PNG: Final = b"\x89PNG\r\n\x1a\n"
### Teams' speaking blue-violet in OpenCV's HSV, inside the adapter's range.
VIOLET: Final = (122, 180, 230)
SIDES: Final = ("top", "left", "right", "bottom")


class FakeStage:
    """The rings told for each frame's instant, none for any other; `seen` records the instants asked."""

    spec: ClassVar[ProviderSpec] = ProviderSpec("fake-stage", Locality.EMBEDDED, Workload.CPU_BOUND, Isolation.LOOP)

    def __init__(self, told: Mapping[float, tuple[Ring, ...]] | None = None) -> None:
        self.told = told or {}
        self.seen: list[float] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def rings(self, frame: Frame, *, width: int, height: int) -> tuple[Ring, ...]:
        self.seen.append(frame.at)
        return self.told.get(frame.at, ())


def strip(tag: int) -> bytes:
    """A name strip as the fake OCR reads one: a PNG header of 8 by 8 and a tag that tells two strips apart."""
    return PNG + bytes(8) + (8).to_bytes(4) + (8).to_bytes(4) + bytes([tag])


def ring(place: tuple[int, int, int, int], tag: int = 0, *, look: int = 0) -> Ring:
    """A ring at `place` whose strip is `strip(tag)` and whose look is four grey bytes of `look`."""
    return Ring(place=place, strip=strip(tag), look=bytes([look]) * 4)


def grid(
    width: int, height: int, *tiles: tuple[tuple[int, int, int, int], tuple[str, ...]], filled: bool = False
) -> bytes:
    """A dark picture of `width` by `height` bgr24 with one ring per tile `((x, y, w, h), sides)`: 4 px sides in the
    speaking colour, the named ones only — or the whole tile `filled` — and a white name band at the tile's foot."""
    colour = cv2.cvtColor(np.array([[VIOLET]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    picture = np.full((height, width, 3), 20, dtype=np.uint8)
    for (x, y, w, h), sides in tiles:
        picture[y + h - 28 : y + h - 2, x + 4 : x + w - 4] = 255
        drawn = {
            "top": (slice(y, y + 4), slice(x, x + w)),
            "bottom": (slice(y + h - 4, y + h), slice(x, x + w)),
            "left": (slice(y, y + h), slice(x, x + 4)),
            "right": (slice(y, y + h), slice(x + w - 4, x + w)),
        }
        picture[y : y + h, x : x + w] = colour if filled else picture[y : y + h, x : x + w]
        for side in sides:
            picture[drawn[side]] = colour
    return picture.tobytes()
