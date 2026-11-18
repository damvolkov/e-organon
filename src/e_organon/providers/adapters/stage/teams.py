"""providers/adapters/stage/teams: the speaking tiles of a Microsoft Teams grid, found in one picture with OpenCV.

Teams rings the tile of whoever is speaking in its brand's blue-violet and writes the person's name in a strip at the
tile's foot. A ring is told from everything else of that colour by shape alone: a large rectangle that is an outline,
not a filled patch, closed on its top, left and right sides — and on its bottom unless the grid's last row runs off
the picture, as it does. A shared screen has no grid and so no ring: the picture answers none, never a guess. The strip
is cut, encoded as PNG for an OCR, and its coarse look kept beside it (grey, 96 by 8) so a caller can tell a strip it
has already read from a new one without reading it again. Pure computation over pixels: off the loop, in a thread."""

from typing import TYPE_CHECKING, ClassVar, Final, Self

import anyio
import cv2
import numpy as np

from e_organon.infra.errors import MediaError
from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload
from e_organon.ports.stage import Ring

if TYPE_CHECKING:
    from e_organon.ports.media import Frame

### the ring's colour in OpenCV's HSV (hue 0-180): Teams' speaking blue-violet, saturated and bright.
LOW: Final = np.array([110, 90, 120], dtype=np.uint8)
HIGH: Final = np.array([135, 255, 255], dtype=np.uint8)
### a tile, not a button: the smallest ring this adapter takes, in pixels of a 1280 by 720 picture.
MIN_WIDTH: Final = 150
MIN_HEIGHT: Final = 90
### an outline: of the ring's bounding box, at most this share is the colour (a filled patch is no ring).
FILLED: Final = 0.25
### a side is drawn when the colour runs along at least this share of it.
SIDE: Final = 0.8
### the thickness, in pixels, a side is looked for in.
EDGE: Final = 3
### the name strip at a tile's foot: this many pixels above its bottom edge, this many below the strip's top, and
### this many in from either side (the ring itself).
STRIP_TOP: Final = 28
STRIP_BOTTOM: Final = 2
STRIP_SIDE: Final = 4
### a tile within this many pixels of the picture's foot runs off it: its ring has no bottom to draw.
FOOT: Final = 2
### a strip's look: small enough to compare cheaply, wide enough to tell two names apart.
LOOK: Final = (96, 8)


##### ADAPTER #####
class TeamsStageAdapter:
    """`StagePort` for Teams recordings: nothing to open, every picture read on its own."""

    spec: ClassVar[ProviderSpec] = ProviderSpec("teams-stage", Locality.EMBEDDED, Workload.CPU_BOUND, Isolation.THREAD)

    __slots__ = ()

    ##### PRIVATE #####

    @staticmethod
    def _rings_drawn(side: np.ndarray, along: int) -> bool:
        """A side is drawn when the colour runs along most of its length (`along`: the axis the length runs on)."""
        return cv2.countNonZero(side.max(axis=1 - along)) > SIDE * side.shape[along]

    @classmethod
    def _rings_closed(cls, mask: np.ndarray, box: tuple[int, int, int, int], height: int) -> bool:
        """Top, left and right drawn, and the bottom too unless the tile runs off the picture."""
        x, y, w, h = box
        top, bottom = mask[y : y + EDGE, x : x + w], mask[y + h - EDGE : y + h, x : x + w]
        left, right = mask[y : y + h, x : x + EDGE], mask[y : y + h, x + w - EDGE : x + w]
        return (
            cls._rings_drawn(top, 1)
            and cls._rings_drawn(left, 0)
            and cls._rings_drawn(right, 0)
            and (y + h >= height - FOOT or cls._rings_drawn(bottom, 1))
        )

    @classmethod
    def _rings_find(cls, pixels: np.ndarray, height: int) -> tuple[tuple[int, int, int, int], ...]:
        """Every closed outline of the speaking colour large enough to be a tile."""
        mask = cv2.inRange(cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV), LOW, HIGH)
        boxes = (
            cv2.boundingRect(contour)
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        )
        return tuple(
            (int(x), int(y), int(w), int(h))
            for x, y, w, h in boxes
            if w >= MIN_WIDTH
            and h >= MIN_HEIGHT
            and cv2.countNonZero(mask[y : y + h, x : x + w]) <= FILLED * w * h
            and cls._rings_closed(mask, (x, y, w, h), height)
        )

    @staticmethod
    def _rings_strip(pixels: np.ndarray, box: tuple[int, int, int, int]) -> Ring:
        """The name strip of one ringed tile, as a PNG and as a look."""
        x, y, w, h = box
        strip = pixels[y + h - STRIP_TOP : y + h - STRIP_BOTTOM, x + STRIP_SIDE : x + w - STRIP_SIDE]
        encoded, png = cv2.imencode(".png", strip)
        match encoded:
            case False:
                msg = f"the name strip of the tile at {box} could not be encoded"
                raise MediaError(msg)
        look = cv2.resize(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), LOOK, interpolation=cv2.INTER_AREA)
        return Ring(place=box, strip=png.tobytes(), look=look.tobytes())

    @classmethod
    def _rings_run(cls, data: bytes, width: int, height: int) -> tuple[Ring, ...]:
        match len(data) == width * height * 3:
            case False:
                msg = f"a picture of {len(data)} bytes is no {width} by {height} bgr24 frame"
                raise MediaError(msg)
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return tuple(cls._rings_strip(pixels, box) for box in cls._rings_find(pixels, height))

    ############################################################

    ##### PUBLIC #####

    async def __aenter__(self) -> Self:
        """Nothing to open: OpenCV is a library."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Nothing to release."""

    async def health(self) -> bool:
        """A library has no heartbeat: present is alive."""
        return True

    async def rings(self, frame: Frame, *, width: int, height: int) -> tuple[Ring, ...]:
        """The tiles marked as speaking in one picture, each with its name strip."""
        return await anyio.to_thread.run_sync(self._rings_run, frame.pixels, width, height)
